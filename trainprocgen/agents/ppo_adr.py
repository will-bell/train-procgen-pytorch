import pathlib
import random
from typing import Union, Dict, List, Tuple, Optional

from procgen import ProcgenEnv
from procgen.domains import DomainConfig, BossfightDomainConfig, datetime_name

from trainprocgen.common.env.procgen_wrappers import *
from trainprocgen.common.misc_util import adjust_lr
from trainprocgen.common.policy import CategoricalPolicy
from .ppo import PPO

Number = Union[int, float]

DEFAULT_DOMAIN_CONFIGS = {
    'dc_bossfight': BossfightDomainConfig()
}


class EnvironmentParameter:

    lower_bound: Number
    upper_bound: Number

    delta: Number

    def __init__(self, name: str, initial_bounds: Tuple[Number, Number], clip_bounds: Tuple[Number, Number], delta: Number, discrete: bool):
        self.name = name
        self.lower_bound, self.upper_bound = initial_bounds
        self.clip_lower_bound, self.clip_upper_bound = clip_bounds
        self.discrete = discrete
        self.delta = delta

    def increase_lower_bound(self):
        self.lower_bound = np.min(self.lower_bound + self.delta, self.upper_bound)

    def decrease_lower_bound(self):
        self.lower_bound = np.max(self.lower_bound - self.delta, self.lower_bound)

    def increase_upper_bound(self):
        self.upper_bound = np.min(self.upper_bound + self.delta, self.clip_upper_bound)

    def decrease_upper_bound(self):
        self.upper_bound = np.max(self.upper_bound - self.delta, self.lower_bound)


class PerformanceBuffer:

    def __init__(self):
        self._buffer = []

    def push_back(self, value: float):
        self._buffer.append(value)

    def is_full(self, size: int) -> bool:
        return len(self._buffer) >= size

    def calculate_average_performance(self) -> float:
        buffer = np.array(self._buffer)
        self.clear()
        return np.mean(buffer).item()

    def clear(self):
        self._buffer = []


class EvaluationConfig:

    n_trajectories: int
    max_buffer_size: int

    def __init__(self, n_trajectories: int = 100, max_buffer_size: int = 10):
        self.n_trajectories = n_trajectories
        self.max_buffer_size = max_buffer_size


class EvaluationEnvironment:

    _env_parameter: EnvironmentParameter

    _train_config_path: pathlib.Path

    _eval_config: EvaluationConfig

    _param_name: str

    _boundary_config_path: str
    _boundary_config: DomainConfig

    _upper_performance_buffer: PerformanceBuffer
    _lower_performance_buffer: PerformanceBuffer

    def __init__(self, env_parameter: EnvironmentParameter, train_config_path: str, eval_config: EvaluationConfig = None):
        self._env_parameter = env_parameter
        self._param_name = self._env_parameter.name
        self._eval_config = eval_config if eval_config is not None else EvaluationConfig()

        self._train_config_path = pathlib.Path(train_config_path)
        config_dir = self._train_config_path.parent
        config_name = self._param_name + '_adr_eval_config.json'

        # Initialize the config for the evaluation environment
        # This config will be updated regularly throughout training. When we boundary sample this environment's
        # parameter, the config will be modified to set the parameter to the selected boundary before running a number
        # of trajectories.
        self._boundary_config = DomainConfig.from_json(self._train_config_path)
        self._boundary_config_path = config_dir / config_name
        self._boundary_config.to_json(self._boundary_config_path)

        # Initialize the environment
        self._env = gym.make('procgen:' + self._boundary_config.game, domain_config_path=str(self._boundary_config_path))

        # Initialize the performance buffers
        self._upper_performance_buffer, self._lower_performance_buffer = PerformanceBuffer(), PerformanceBuffer()

    def evaluate_performance(self, policy: CategoricalPolicy, hidden_state: np.ndarray) -> Optional[Tuple[float, bool]]:
        """Main method for running the ADR algorithm

        Args:
            policy:
            hidden_state:

        Returns:

        """
        # Load the current training config to get any changes to the environment parameters
        updated_train_config = DomainConfig.from_json(self._train_config_path)
        updated_params = updated_train_config.parameters

        x = random.uniform(0., 1.)
        if x < .5:
            lower = True
            value = self._env_parameter.lower_bound
            buffer = self._lower_performance_buffer
        else:
            lower = False
            value = self._env_parameter.upper_bound
            buffer = self._upper_performance_buffer

        updated_params['min_' + self._param_name] = value
        updated_params['max_' + self._param_name] = value
        self._boundary_config.update_parameters(updated_params, cache=False)

        self._generate_trajectories(policy, hidden_state, buffer)

        if buffer.is_full(self._eval_config.max_buffer_size):
            performance = buffer.calculate_average_performance()
            return performance, lower

        return None

    def update_lower_bound(self, performance: float, thresholds: Tuple[float, float]) -> Number:
        low_threshold, high_threshold = thresholds
        if performance >= high_threshold:
            self._env_parameter.increase_lower_bound()
        elif performance <= low_threshold:
            self._env_parameter.decrease_lower_bound()

        return self._env_parameter.lower_bound

    def update_upper_bound(self, performance: float, thresholds: Tuple[float, float]) -> Number:
        low_threshold, high_threshold = thresholds
        if performance >= high_threshold:
            self._env_parameter.increase_upper_bound()
        elif performance <= low_threshold:
            self._env_parameter.decrease_upper_bound()

        return self._env_parameter.upper_bound

    @staticmethod
    def _predict(policy, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs)
            hidden_state = torch.FloatTensor(hidden_state)
            mask = torch.FloatTensor(1 - done)
            dist, value, hidden_state = policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy()

    def _generate_trajectories(self, policy: CategoricalPolicy, hidden_state: np.ndarray, buffer: PerformanceBuffer):
        obs = self._env.reset()
        rewards = []
        last_steps = []
        for _ in range(self._eval_config.n_trajectories):
            done = False
            while not done:
                act, _, _, next_hidden_state = self._predict(policy, obs, hidden_state, done)
                next_obs, rew, done, _ = self._env.step(act)
                obs = next_obs
                hidden_state = next_hidden_state

                rewards.append(rew)
                last_steps.append(done)

            obs = self._env.reset()

        mean_return = self._calculate_average_return(rewards, last_steps)
        buffer.push_back(mean_return)

    @staticmethod
    def _calculate_average_return(rewards: List[float], last_steps: List[bool]) -> float:
        G = rewards[-1]
        returns = []
        for i in reversed(range(len(rewards))):
            rew = rewards[i]
            done = last_steps[i]

            G = rew + .99 * G * (1 - done)
            returns.append(G)

        returns = np.array(returns)

        return np.mean(returns).item()


def make_environments(env_name: str,
                      initial_domain_config: DomainConfig = None,
                      tunable_parameters: List['str'] = None,
                      experiment_dir: Union[pathlib.Path, str] = None,
                      eval_config: EvaluationConfig = None,
                      n_training_envs: int = 8):

    if initial_domain_config is None:
        try:
            initial_domain_config = DEFAULT_DOMAIN_CONFIGS[env_name]
        except KeyError:
            raise KeyError(f'No default config exists for {env_name}')

    if experiment_dir is None:
        experiment_dir = pathlib.Path().absolute() / 'adr_experiments' / 'experiment-' + datetime_name()
        experiment_dir.mkdir(parents=True, exist_ok=False)
    else:
        experiment_dir = pathlib.Path(experiment_dir)

    config_dir = experiment_dir / 'domain_configs'
    train_config_path = config_dir / 'train_config.json'

    initial_domain_config.to_json(train_config_path)

    torch.set_num_threads(1)
    training_env = ProcgenEnv(num_envs=n_training_envs,
                              env_name=env_name,
                              domain_config=train_config_path)
    training_env = VecExtractDictObs(training_env, "rgb")
    training_env = TransposeFrame(training_env)
    training_env = ScaledFloatFrame(training_env)

    evaluation_envs = {}
    for param in tunable_parameters:
        evaluation_envs[param] = EvaluationEnvironment(param, train_config_path, eval_config)

    return training_env, initial_domain_config, evaluation_envs


class PPOADR(PPO):

    _train_domain_config: DomainConfig
    _evaluation_envs: Dict[str, EvaluationEnvironment]
    _performance_thresholds: Tuple[float, float]

    def __init__(self,
                 training_env,
                 initial_domain_config,
                 evaluation_envs,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=32 * 8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 adr_prob: float = 0.5,
                 performance_thresholds: Tuple[float, float] = (2.5, 8.),
                 **kwargs):

        super().__init__(training_env, policy, logger, storage, device, n_checkpoints, n_steps, n_envs, epoch,
                         mini_batch_per_epoch, mini_batch_size, gamma, lmbda, learning_rate, grad_clip_norm, eps_clip,
                         value_coef, entropy_coef, normalize_adv, normalize_rew, use_gae, **kwargs)

        self._train_domain_config = initial_domain_config
        self._evaluation_envs = evaluation_envs
        self._tunable_params = list(evaluation_envs.keys())
        self._n_tunable_params = len(self._tunable_params)
        self._adr_prob = adr_prob
        self._performance_thresholds = performance_thresholds

        self._obs = None
        self._hidden_state = None
        self._done = None

    def _generate_training_data(self):
        for _ in range(self.n_steps):
            act, log_prob_act, value, next_hidden_state = self.predict(self._obs, self._hidden_state, self._done)
            next_obs, rew, done, info = self.env.step(act)
            self.storage.store(self._obs, self._hidden_state, act, rew, self._done, info, log_prob_act, value)
            self._obs = next_obs
            self._hidden_state = next_hidden_state
        _, _, last_val, self._hidden_state = self.predict(self._obs, self._hidden_state, self._done)
        self.storage.store_last(self._obs, self._hidden_state, last_val)

    def _evaluate_performance(self):
        # Randomly select a parameter to boundary sample
        param_idx = random.randint(self._n_tunable_params)
        param_name = self._tunable_params[param_idx]

        # Get the environment for the selected parameter then evaluate the policy within it. This will boundary sample
        # the selected parameter and generate a number of trajectories with the upper/lower boundary to calculate its
        # performance in the environment.
        evaluation_env = self._evaluation_envs[param_name]
        info = evaluation_env.evaluate_performance(self.policy, self._hidden_state)

        # If we get something back, then the performance buffer for either the lower or upper boundary of the parameter
        # is filled. Update the parameter according to the set thresholds.
        if info:
            performance, lower = info
            if lower:  # Updating the lower boundary according to set performance thresholds
                prefix = 'min_'
                new_value = evaluation_env.update_lower_bound(performance, self._performance_thresholds)
            else:      # Updating the upper boundary according to set performance thresholds
                prefix = 'max_'
                new_value = evaluation_env.update_upper_bound(performance, self._performance_thresholds)

            # Update the config for the training environment with the new value for the boundary-sampled parameter.
            self._train_domain_config.update_parameters({prefix + param_name: new_value})

    def train(self, num_timesteps: int):
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        self._obs = self.env.reset()
        self._hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        self._done = np.zeros(self.n_envs)

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()

            x = random.uniform(0., 1.)
            if x < self._adr_prob and self._hidden_state:
                self._evaluate_performance()
            else:
                self._generate_training_data()

            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # Update policy & values
            summary = self.update_policy()

            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()
            self.logger.feed(rew_batch, done_batch)
            self.logger.write_summary(summary)
            self.logger.dump()
            self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)

            # Save the model
            if self.t > ((checkpoint_cnt + 1) * save_every):
                torch.save({'state_dict': self.policy.state_dict()},
                           self.logger.logdir + '/model_' + str(self.t) + '.pth')
                checkpoint_cnt += 1

        self.env.close()
