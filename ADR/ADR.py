"""
Implementation of Automatic Domain Randomization (https://arxiv.org/pdf/1910.07113.pdf)
"""

import torch
import gym
from procgen.domains import DomainConfig
import pathlib
from typing import Union, Tuple, List, Optional
import random
from torch import nn
import numpy as np
from trainprocgen.common.policy import CategoricalPolicy
from trainprocgen.common.storage import Storage

import os

MAX_SIZE_BUFFER = 10

Number = Union[int, float]


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


class ADRManager:
    def __init__(self, parameters_list: list, policy: CategoricalPolicy, n_envs: int, hidden_state_size: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.parameters_list = parameters_list
        self.parameters = {}
        for param in parameters_list:
            self.parameters[param.name] = param

        column_names = [param.name + '_low' for param in self.parameters_list] \
            + [param.name + '_hi' for param in self.parameters_list]
        self.result_dataframe = pd.DataFrame(columns=column_names)
        self.running_dataframes = []

        self.num_trajectory = MAX_SIZE_BUFFER  # idk change this later? Do all trajectories at once??
        self.policy = policy
        self.n_envs = n_envs
        self.hidden_state_size = hidden_state_size

    def predict(self, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy()

    def evaluate_performance(self, env):
        # Call self.append_performance here too
        done = False

        average_returns = []
        for trajectory in range(self.num_trajectory):
            obs = env.reset()
            obs = obs['rgb']
            hidden_state = np.zeros((self.n_envs, self.hidden_state_size))
            done = np.zeros(self.n_envs)

            value_batch = []
            rew_batch = []
            done_batch = []

            for i in range(20):
                act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done)
                next_obs, rew, done, info = env.step(act)
                next_obs = next_obs['rgb']
                # self.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
                value_batch.append(value)
                rew_batch.append(rew)
                done_batch.append(done)
                obs = next_obs
                hidden_state = next_hidden_state

            return_batch = self.compute_estimates(value_batch, rew_batch, done_batch)
            average_return = np.mean(return_batch)
            average_returns.append(average_return)

        return np.mean(average_returns)

    def compute_estimates(self, value_batch: list, rew_batch: list, done_batch: list, gamma=0.99):
        return_batch = [0] * len(value_batch)
        G = value_batch[-1]
        for i in reversed(range(len(value_batch))):
            rew = rew_batch[i]
            done = done_batch[i]

            G = rew + gamma * G * (1 - done)
            return_batch[i] = G

        return return_batch

    def get_environment_parameters(self) -> Dict[str, ADREnvParameter]:
        return self.parameters

    def append_performance(self, feature_ind: int, is_high: bool, performance: float):
        """Append performance to performance buffer of boundary sampled feature

        Args:
            feature_ind (int): feature index
            is_high (bool): high or low flag for choosing between phi_L or phi_H
            performance (float): performance calculation
        """
        reached_max_buffer = self.parameters_list[feature_ind] \
            .get_param(is_high) \
            .append_performance(performance)

        if reached_max_buffer:
            d = dict()
            for param in self.parameters:
                d[param.name + '_low'] = param.get_param(is_high=False).value
                d[param.name + '_hi'] = param.get_param(is_high=True).value

            d['performance'] = performance
            self.running_dataframes.append(d)

            # Add new dataframes to running list of dataframes
            # Either concat now or write to csv
            if len(self.running_dataframes) >= 10:
                self.running_dataframes = pd.DataFrame(self.running_dataframes)
                # frames = [self.result_dataframe, self.running_dataframes]
                # self.result_dataframe = pd.concat(frames)

                # If result file already exists, append to the file
                # Or else write a new file
                if os.path.exists('results.csv'):
                    self.running_dataframes.to_csv('results.csv', mode='a', header=False)
                else:
                    self.running_dataframes.to_csv('results.csv')

                # Reset dataframes
                self.running_dataframes = []

    def select_boundary_sample(self):
        """ Selects feature index to boundary sample. Also uniformly selects a probability
        between 0 < x < 1 for low and high of phi

        Returns:
            int: feature index
            float: probability to choose between phi_L and phi_H
        """

        # Reset adr flag
        for param in self.parameters:
            self.parameters[param].set_adr_flag(False)

        feature_to_boundary_sample = torch.randint(0, len(self.parameters_list), size=(1,)).item()
        self.parameters_list[feature_to_boundary_sample].set_adr_flag(True)

        probability = torch.rand(1).item()  # 0 <= x < 1

        return feature_to_boundary_sample, probability

    def create_config(self, feature_to_boundary_sample: int, probability: float):
        config = {}

        # TODO how to handle append_performance in ADRParameter ??
        for i, env_parameter in enumerate(self.parameters_list):
            _lambda = env_parameter.sample(probability)
            if i == feature_to_boundary_sample:
                # boundary_sample returns ADRParameter, so call return_val to get its value
                _lambda = _lambda.return_val()

            config[env_parameter.name] = _lambda
        return config

    def adr_entropy(self):
        """ Calculate ADR Entrophy

        Returns:
            float: entropy =  1/d \sum_{i=1}^{d} log(phi_ih - phi_il)
        """
        d = len(self.parameters_list)
        phi_H = []
        phi_L = []

        for i in range(d):
            phi_H.append(self.parameters_list[i].phi_h.return_val())
            phi_L.append(self.parameters_list[i].phi_l.return_val())

        phi_H = torch.tensor(phi_H, dtype=torch.float)
        phi_L = torch.tensor(phi_L, dtype=torch.float)

        entropy = torch.mean(torch.log(phi_H - phi_L))
        return entropy