# -*- coding: utf-8 -*-
import gym
import math
import random
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_rl.replay import ReplayMemory
from pytorch_rl.utils import soft_update
from attrdict import AttrDict
import numpy as np


class _MLP(nn.Module):
    def __init__(self):
        super(_MLP, self).__init__()
        self.fc1 = nn.Linear(4, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.head = nn.Linear(200, 2)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        return self.head(x)


DEFAULT_CONFIG = AttrDict({
    'batch_size': 128,
    'gamma': 0.999,
    'eps_start': 0.9,
    'eps_end': 0.05,
    'eps_decay': 200
})


class DQN(object):
    def __init__(self, env, args=DEFAULT_CONFIG):
        self.env = env
        self.args = args

        self.model = _MLP()
        self.target_model = _MLP()
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.RMSprop(self.model.parameters())
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def act(self, state):
        model = self.model
        model.eval()

        sample = random.random()
        args = self.args
        eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
                        math.exp(-1. * self.steps_done / args.eps_decay)
        if sample > eps_threshold:
            return model(state.unsqueeze(0)).data.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]]).long()

    def optimize_model(self):
        self.model.train()
        args = self.args

        if len(self.memory) < args.batch_size:
            return

        batch = self.memory.sample(args.batch_size)
        #  the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(list(map(lambda s: s is not None,
                                               batch.next_state))).byte()
        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None], 0)

        state_batch = torch.stack(batch.state, 0)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        # Q(s, a)
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(args.batch_size)

        next_action_batch = self.model(non_final_next_states).max(1)[1].unsqueeze(1)
        # next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0]
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).gather(1,
                                                                                            next_action_batch).squeeze()
        next_state_values.detach_()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * args.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), 1)
        self.optimizer.step()
        self.steps_done += 1
        soft_update(self.target_model, self.model, .9)

    def run(self, num_episodes=1000000):
        from torchnet.meter import AverageValueMeter
        from pytorch_rl.utils.progress_bar import json_progress_bar

        reward_meter = AverageValueMeter()
        progress_bar = json_progress_bar(range(num_episodes), prefix='training')
        for i_episode in progress_bar:
            # Initialize the environment and state
            obs = torch.from_numpy(env.reset()).float()
            episode_rewards = 0
            for t in count():
                # Select and perform an action
                action = self.act(obs.float())
                env.render()

                next_obs, reward, done, _ = env.step(action[0, 0].item())
                next_obs = torch.from_numpy(next_obs).float()

                episode_rewards += reward
                reward = torch.from_numpy(np.array([reward])).float()

                if not done:
                    next_obs = next_obs
                else:
                    next_obs = None

                # Store the transition in memory
                self.memory.push(obs, action, next_obs, done, reward)

                # Move to the next state
                obs = next_obs

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    break

            reward_meter.add(episode_rewards)
            from collections import OrderedDict
            stats = OrderedDict(episode=i_episode,
                                reward=reward_meter)
            progress_bar.log(stats)
            if i_episode % 10 == 0:
                progress_bar.print(stats)
            # if i_episode % 10 == 0:
            #     progress_bar.print('i_episode: {}, reward: {} (+/- {})'.format(i_episode, reward_meter.mean,
            #                                                                    reward_meter.std))


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # if gpu is to be used
    use_cuda = torch.cuda.is_available()
    env.seed(42)
    env.reset()
    DQN(env).run()
