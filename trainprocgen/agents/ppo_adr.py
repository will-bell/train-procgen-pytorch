from typing import Union, Dict
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from trainprocgen.common.misc_util import adjust_lr
from .ppo import PPO
from procgen.domains import DomainConfig

from ADR import ADRParameter, ADREnvParameter

Number = Union[int, float]


class ADRManager:
    def __init__(self, parameters_list: list):
        self.parameters_list = parameters_list
        self.parameters = {}
        for param in parameters_list:
            self.parameters[param.name] = param
        
        column_names = [param.name + '_low' for param in self.parameters_list] \
                    + [param.name + '_hi' for param in self.parameters_list]
        self.result_dataframe = pd.DataFrame(columns=column_names)
        self.running_dataframes = []
    
    def evaluate_performance(self, policy: nn.Module):
        # Call self.append_performance here too
        pass
    
    def get_environment_parameters(self) -> Dict[str, ADREnvParameter]:
        return self.parameters
    
    def append_performance(self, feature_ind: int, is_high: bool, performance: float):
        """Append performance to performance buffer of boundary sampled feature

        Args:
            feature_ind (int): feature index
            is_high (bool): high or low flag for choosing between phi_L or phi_H
            performance (float): performance calculation
        """
        reached_max_buffer = self.parameters_list[feature_ind]\
                                    .get_param(is_high)\
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
            if len(self.running_dataframes >= 10):
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
        
        probability = torch.rand(1).item() # 0 <= x < 1

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

class PPOADR(PPO):

    def __init__(self,
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 domain_config: DomainConfig,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=32*8,
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
                 **kwargs):

        super().__init__(env, policy, logger, storage, device, n_checkpoints, n_steps, n_envs, epoch,
                         mini_batch_per_epoch, mini_batch_size, gamma, lmbda, learning_rate, grad_clip_norm, eps_clip,
                         value_coef, entropy_coef, normalize_adv, normalize_rew, use_gae, **kwargs)

        self.adr_prob = adr_prob
        self.adr_manager = ADRManager()
        self.domain_config = domain_config

    def train(self, num_timesteps: int):
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()

            self.env.reset()

            for _ in range(self.n_steps):
                act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done)
                next_obs, rew, done, info = self.env.step(act)
                self.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
                obs = next_obs
                hidden_state = next_hidden_state
            _, _, last_val, hidden_state = self.predict(obs, hidden_state, done)
            self.storage.store_last(obs, hidden_state, last_val)

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
