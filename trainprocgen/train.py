import argparse
import random

import yaml
from procgen import ProcgenEnv

from trainprocgen.common import set_global_seeds, set_global_log_levels
from trainprocgen.common.env.procgen_wrappers import *
from trainprocgen.common.logger import Logger
from trainprocgen.common.model import NatureModel, ImpalaModel
from trainprocgen.common.policy import CategoricalPolicy
from trainprocgen.common.storage import Storage


def procgen_train_function(
        env_name: str = 'bossfight',
        exp_name: str = 'test',
        n_steps: int = 256,
        n_envs: int = 64,
        architecture: str = 'impala',
        recurrent: bool = False,
        algorithm: str = 'ppo',
        normalize_rew: bool = True,
        logdir: str = None):

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    # DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ENVIRONMENT
    print('INITIALIZAING ENVIRONMENTS...')
    # By default, pytorch utilizes multi-threaded cpu
    # Procgen is able to handle thousand of steps on a single core
    torch.set_num_threads(1)
    env = ProcgenEnv(num_envs=n_envs,
                     env_name=env_name,
                     start_level=start_level,
                     num_levels=num_levels,
                     distribution_mode=distribution_mode)
    env = VecExtractDictObs(env, "rgb")
    if normalize_rew:
        env = VecNormalize(env, ob=False)  # normalizing returns, but not the img frames.
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)

    # LOGGER
    print('INITIALIZAING LOGGER...')
    if logdir is None:
        logdir = 'procgen/' + env_name + '/' + exp_name + '/' + 'seed' + '_' + \
                 str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('logs', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logger = Logger(n_envs, logdir)

    # MODEL
    print('INTIALIZING MODEL...')
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    in_channels = observation_shape[0]
    action_space = env.action_space

    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)
    else:
        raise NotImplementedError('Only Nature and Impala models are implemented')

    # Discrete action space
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size)
    else:
        raise NotImplementedError
    policy.to(device)

    # STORAGE
    print('INITIALIZAING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)

    # AGENT
    print('INTIALIZING AGENT...')
    if algorithm == 'ppo':
        from trainprocgen.agents.ppo import PPO as AGENT
    else:
        raise NotImplementedError('Only PPO is implemented')
    agent = AGENT(env, policy, logger, storage, device, num_checkpoints, **hyperparameters)

    # TRAINING
    print('START TRAINING...')
    agent.train(num_timesteps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',          type=str, default='test', help='experiment name')
    parser.add_argument('--env_name',          type=str, default='starpilot', help='environment ID')
    parser.add_argument('--start_level',       type=int, default=int(0), help='start-level for environment')
    parser.add_argument('--num_levels',        type=int, default=int(0), help='number of training levels for environment')
    parser.add_argument('--distribution_mode', type=str, default='easy', help='distribution mode for environment')
    parser.add_argument('--param_name',        type=str, default='easy-200', help='hyper-parameter ID')
    parser.add_argument('--gpu_device',        type=int, default=int(0), required=False, help='visible device in CUDA')
    parser.add_argument('--num_timesteps',     type=int, default=int(25000000), help='number of training timesteps')
    parser.add_argument('--seed',              type=int, default=random.randint(0, 9999), help='Random generator seed')
    parser.add_argument('--log_level',         type=int, default=int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints',   type=int, default=int(1), help='number of checkpoints to store')
    parser.add_argument('--log_dir',           type=str, default=None, required=False)

    args = parser.parse_args()
    _exp_name = args.exp_name
    _env_name = args.env_name
    start_level = args.start_level
    num_levels = args.num_levels
    distribution_mode = args.distribution_mode
    param_name = args.param_name
    num_timesteps = args.num_timesteps
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints
    log_dir = args.log_dir

    print('[LOADING HYPERPARAMETERS...]')
    with open('hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    for key, value in hyperparameters.items():
        print(key, ':', value)

    _n_steps = hyperparameters.get('n_steps', 256)
    _n_envs = hyperparameters.get('n_envs', 64)
    _architecture = hyperparameters.get('architecture', 'impala')
    _recurrent = hyperparameters.get('recurrent', False)
    algo = hyperparameters.get('algo', 'ppo')
    _normalize_rew = hyperparameters.get('normalize_rew', True)

    procgen_train_function(_env_name, _exp_name, _n_steps, _n_envs, _architecture, _recurrent, algo, _normalize_rew, log_dir)
