from typing import Union, Dict
import random

import numpy as np
import torch
from torch import nn

from trainprocgen.common.misc_util import adjust_lr
from .ppo import PPO
from procgen.domains import DomainConfig

Number = Union[int, float]


class ADRManager:

    def __init__(self):
        pass

    def get_environment_parameters(self) -> Dict[str, Number]:
        pass


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

        self._hidden_state = None

    def _generate_training_data(self):
        obs = self.env.reset()
        done = np.zeros(self.n_envs)
        for _ in range(self.n_steps):
            act, log_prob_act, value, next_hidden_state = self.predict(obs, self._hidden_state, done)
            next_obs, rew, done, info = self.env.step(act)
            self.storage.store(obs, self._hidden_state, act, rew, done, info, log_prob_act, value)
            obs = next_obs
            self._hidden_state = next_hidden_state
        _, _, last_val, hidden_state = self.predict(obs, self._hidden_state, done)
        self.storage.store_last(obs, hidden_state, last_val)

    def train(self, num_timesteps: int):
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        self._hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()

            x = random.uniform(0., 1.)
            if x < self.adr_prob:
                self._generate_training_data()
            else:
                pass

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
