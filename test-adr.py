from ADR import *
from bossfight_setup import BossfightDomainConfig, make_interactive_bossfight_env

import matplotlib.pyplot as plt
import gym3

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
    
    manager = ADRManager(parameters)
    feature_to_boundary_sample, probability = manager.select_boundary_sample()
    
    config = manager.create_config(feature_to_boundary_sample, probability)
    
    print(config)
    test_config = BossfightDomainConfig(**config)
    env = make_interactive_bossfight_env(test_config)

    h, w, _ = env.ob_space["rgb"].shape
    
    step = 0
    for i in range(1000):
        env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, first = env.observe()
        print(f"step {step} reward {rew} first {first}")
        step += 1

    # # Testing to see if torch.randint is Uniformly Distributed
    # samples = [torch.randint(0, len(parameters), size=(1,)).item() for i in range(100000)]
    # plt.hist(samples, bins=len(parameters),)
    # plt.show()
