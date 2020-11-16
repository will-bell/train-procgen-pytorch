from ADR import *
from trainprocgen.agents import ppo_adr

from bossfight_setup import BossfightDomainConfig, make_interactive_bossfight_env
from trainprocgen.common.model import NatureModel, ImpalaModel
from trainprocgen.common.policy import CategoricalPolicy
from trainprocgen.common.storage import Storage

import matplotlib.pyplot as plt
import gym3, gym

"""
Default configurable settings for Bossfight
min_n_rounds=2, 
max_n_rounds=2, 
min_n_barriers: 1, 
max_n_barriers: 4, 
min_boss_round_health: 2, 
max_boss_round_health: 2, 
min_boss_invulnerable_duration: 3, 
max_boss_invulnerable_duration: 3, 
n_boss_attack_modes: 4, 
min_boss_bullet_velocity: 0.5, 
max_boss_bullet_velocity: 0.75, 
boss_rand_fire_prob: 0.1, 
boss_scale: 0.5
"""
if __name__ == '__main__':
    min_n_rounds = ADREnvParameter(name='min_n_rounds', value=5, lower_bound=1, upper_bound=10, step_size=1, thresh_low=2.5, thresh_high=7.5, is_continuous=False) 
    max_n_rounds = ADREnvParameter(name='max_n_rounds', value=5, lower_bound=1, upper_bound=10, step_size=1, thresh_low=2.5, thresh_high=7.5, is_continuous=False)
    min_n_barriers = ADREnvParameter(name='min_n_barriers', value=1, lower_bound=0, upper_bound=25, step_size=1, thresh_low=2.5, thresh_high=7.5, is_continuous=False)
    max_n_barriers = ADREnvParameter(name='max_n_barriers', value=4, lower_bound=1, upper_bound=25, step_size=1, thresh_low=2.5, thresh_high=7.5, is_continuous=False)
    min_boss_round_health = ADREnvParameter(name='min_boss_round_health', value=2, lower_bound=1, upper_bound=25, step_size=1, thresh_low=2.5, thresh_high=7.5, is_continuous=False) 
    max_boss_round_health = ADREnvParameter(name='max_boss_round_health', value=2, lower_bound=1, upper_bound=25, step_size=1, thresh_low=2.5, thresh_high=7.5, is_continuous=False)  
    min_boss_invulnerable_duration = ADREnvParameter(name='min_boss_invulnerable_duration', value=3, lower_bound=0, upper_bound=25, step_size=1, thresh_low=2.5, thresh_high=7.5, is_continuous=False) 
    max_boss_invulnerable_duration = ADREnvParameter(name='max_boss_invulnerable_duration', value=3, lower_bound=0, upper_bound=25, step_size=1, thresh_low=2.5, thresh_high=7.5, is_continuous=False) 
    n_boss_attack_modes = ADREnvParameter(name='n_boss_attack_modes', value=4, lower_bound=1, upper_bound=25, step_size=1, thresh_low=2.5, thresh_high=7.5, is_continuous=True) 
    min_boss_bullet_velocity = ADREnvParameter(name='min_boss_bullet_velocity', value=0.5, lower_bound=0.10, upper_bound=10, step_size=0.1, thresh_low=2.5, thresh_high=7.5, is_continuous=True)
    max_boss_bullet_velocity = ADREnvParameter(name='max_boss_bullet_velocity', value=0.75, lower_bound=0.5, upper_bound=25, step_size=0.1, thresh_low=2.5, thresh_high=7.5, is_continuous=True) 
    boss_rand_fire_prob = ADREnvParameter(name='boss_rand_fire_prob', value=0.05, lower_bound=0.01, upper_bound=1, step_size=0.01, thresh_low=2.5, thresh_high=7.5, is_continuous=True) 
    boss_scale = ADREnvParameter(name='boss_scale', value=0.5, lower_bound=0.1, upper_bound=1, step_size=0.1, thresh_low=2.5, thresh_high=7.5, is_continuous=True) 
    
    parameters = [min_n_rounds, 
                max_n_rounds, 
                min_n_barriers, 
                max_n_barriers, 
                min_boss_round_health, 
                max_boss_round_health, 
                min_boss_invulnerable_duration, 
                max_boss_invulnerable_duration, 
                n_boss_attack_modes, 
                min_boss_bullet_velocity, 
                max_boss_bullet_velocity, 
                boss_rand_fire_prob, 
                boss_scale]
    
    num_parameters = len(parameters)
    print(f'Number of parameters (d): num_parameters')
    # feature_to_boundary_sample = torch.randint(0, len(parameters), size=(1,)).item()
    # parameters[feature_to_boundary_sample].set_adr_flag(True)
    
    n_envs = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MODEL
    print('INTIALIZING MODEL...')
    env = make_interactive_bossfight_env(None, n_envs=n_envs)

    observation_space = env.observation_space
    print(observation_space)
    print(observation_space['rgb'].shape)
    observation_shape = observation_space['rgb'].shape
    print(observation_shape)
    in_channels = observation_shape[0]
    action_space = env.action_space
    print(action_space)
    
    architecture = 'impala'
    recurrent = True
    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels, input_shape=observation_shape)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels, input_shape=observation_shape)
    else:
        raise NotImplementedError('Only Nature and Impala models are implemented')

    # Discrete action space
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size)
    else:
        raise NotImplementedError
    policy.to(device)
    hidden_state_size = model.output_dim

    manager = ppo_adr.ADRManager(parameters, policy, n_envs, hidden_state_size)
    feature_to_boundary_sample, probability = manager.select_boundary_sample()
    is_high = probability >= 0.5
    config = manager.create_config(feature_to_boundary_sample, probability)
    
    print(config)
    test_config = BossfightDomainConfig(**config)
    env = make_interactive_bossfight_env(test_config, n_envs=n_envs)

    performance = manager.evaluate_performance(env)
    manager.append_performance(feature_to_boundary_sample,
                               is_high,
                               performance)
