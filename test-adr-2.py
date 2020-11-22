from trainprocgen.agents import *
from trainprocgen.common.model import NatureModel, ImpalaModel
from trainprocgen.common.policy import CategoricalPolicy
from trainprocgen.common.storage import Storage
from trainprocgen.common.logger import Logger

import torch

if __name__ == '__main__':
    torch.set_num_threads(1)
    
    n_rounds = EnvironmentParameter(name='n_rounds', initial_bounds=(5,5), clip_bounds=(1,10), delta=1, discrete=True) 
    n_barriers = EnvironmentParameter(name='n_barriers', initial_bounds=(1, 1), clip_bounds=(1,25), delta=1, discrete=True)
    boss_round_health = EnvironmentParameter(name='boss_round_health', initial_bounds=(2,2), clip_bounds=(1,25), delta=1, discrete=True) 
    boss_invulnerable_duration = EnvironmentParameter(name='boss_invulnerable_duration', initial_bounds=(3,3), clip_bounds=(0,25), delta=1, discrete=True) 
    n_boss_attack_modes = EnvironmentParameter(name='n_boss_attack_modes', initial_bounds=(4,4), clip_bounds=(1,25), delta=1, discrete=False) 
    boss_bullet_velocity = EnvironmentParameter(name='boss_bullet_velocity', initial_bounds=(0.5,0.5), clip_bounds=(0.1,10), delta=0.1, discrete=False)
    boss_rand_fire_prob = EnvironmentParameter(name='boss_rand_fire_prob', initial_bounds=(0.05,0.05), clip_bounds=(0.01,1), delta=0.01, discrete=False) 
    boss_scale = EnvironmentParameter(name='boss_scale', initial_bounds=(0.5,0.5), clip_bounds=(0.01,1), delta=0.1, discrete=False) 
    
    tunable_parameters = [
                n_rounds, 
                n_barriers, 
                boss_round_health, 
                boss_invulnerable_duration, 
                n_boss_attack_modes, 
                boss_bullet_velocity, 
                boss_rand_fire_prob, 
                boss_scale]
    n_envs = 64
    num_steps = 256
    eval_config = EvaluationConfig(n_trajectories=100, max_buffer_size=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_env, initial_domain_config, evaluation_envs = make_environments(env_name = 'dc_bossfight',
                                                                            initial_domain_config = None,
                                                                            tunable_parameters= tunable_parameters,
                                                                            experiment_dir = None,
                                                                            eval_config = eval_config,
                                                                            n_training_envs = n_envs,
                                                                            device = device)
    
    print(training_env)
    # print(initial_domain_config)

    observation_space = training_env.observation_space
    observation_shape = observation_space.shape
    print(f'initial obs shape: {observation_shape}')

    in_channels = observation_shape[0]
    action_space = training_env.action_space

    print(f'in channels: {in_channels}')
    model = ImpalaModel(in_channels=in_channels, input_shape=observation_shape).to(device)
    action_size = action_space.n
    recurrent = True
    policy = CategoricalPolicy(model, recurrent, action_size)
    if recurrent:
        policy.gru.to(device)
    policy.to(device)
    
    logger = Logger(n_envs=n_envs, logdir='./log')

    hidden_state_size = model.output_dim
    storage = Storage(observation_shape, hidden_state_size, num_steps=num_steps, num_envs=n_envs, device=device)
    n_checkpoints = 100
    ppo_adr = PPOADR(training_env,
                    initial_domain_config,
                    evaluation_envs,
                    policy,
                    logger,
                    storage,
                    device,
                    n_checkpoints,
                    n_steps=num_steps,
                    n_envs=n_envs,
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
                    use_gae=False,
                    adr_prob = 0.5,
                    performance_thresholds = (2.5, 8.)
                    )
    
    ppo_adr.train(200000000)
