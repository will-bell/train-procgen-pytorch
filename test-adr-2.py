from trainprocgen.agents import *

if __name__ == '__main__':
    min_n_rounds = EnvironmentParameter(name='min_n_rounds', initial_bounds=(5,5), clip_bounds=(1,10), delta=1, discrete=True) 
    max_n_rounds = EnvironmentParameter(name='max_n_rounds', initial_bounds=(5,5), clip_bounds=(1,10), delta=1, discrete=True)
    min_n_barriers = EnvironmentParameter(name='min_n_barriers', initial_bounds=(1, 1), clip_bounds=(0,25), delta=1, discrete=True)
    max_n_barriers = EnvironmentParameter(name='max_n_barriers', initial_bounds=(4,4), clip_bounds=(1,25), delta=1, discrete=True)
    min_boss_round_health = EnvironmentParameter(name='min_boss_round_health', initial_bounds=(2,2), clip_bounds=(1,25), delta=1, discrete=True) 
    max_boss_round_health = EnvironmentParameter(name='max_boss_round_health', initial_bounds=(2,2), clip_bounds=(1,25), delta=1, discrete=True)  
    min_boss_invulnerable_duration = EnvironmentParameter(name='min_boss_invulnerable_duration', initial_bounds=(3,3), clip_bounds=(0,25), delta=1, discrete=True) 
    max_boss_invulnerable_duration = EnvironmentParameter(name='max_boss_invulnerable_duration', initial_bounds=(3,3), clip_bounds=(0,25), delta=1, discrete=True) 
    n_boss_attack_modes = EnvironmentParameter(name='n_boss_attack_modes', initial_bounds=(4,4), clip_bounds=(1,25), delta=1, discrete=False) 
    min_boss_bullet_velocity = EnvironmentParameter(name='min_boss_bullet_velocity', initial_bounds=(0.5,0.5), clip_bounds=(0.1,10), delta=0.1, discrete=False)
    max_boss_bullet_velocity = EnvironmentParameter(name='max_boss_bullet_velocity', initial_bounds=(0.75,0.75), clip_bounds=(0.5,25), delta=0.1, discrete=False) 
    boss_rand_fire_prob = EnvironmentParameter(name='boss_rand_fire_prob', initial_bounds=(0.05,0.05), clip_bounds=(0.01,1), delta=0.01, discrete=False) 
    boss_scale = EnvironmentParameter(name='boss_scale', initial_bounds=(0.5,0.5), clip_bounds=(0.01,1), delta=0.1, discrete=False) 
    
    tunable_parameters = [min_n_rounds, 
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
    
    
    eval_config = EvaluationConfig(n_trajectories=100, max_buffer_size=10)
    training_env, initial_domain_config, evaluation_envs = make_environments(env_name = 'dc_bossfight',
                                                                            initial_domain_config = None,
                                                                            tunable_parameters= tunable_parameters,
                                                                            experiment_dir = None,
                                                                            eval_config = eval_config,
                                                                            n_training_envs = 8)