import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from pytorch_rl.ounoise import OUNoise
from pytorch_rl.replay import ReplayMemory
from pytorch_rl.cdqn import DEFAULT_CONFIG
from pytorch_rl.utils import soft_update, hard_update
from pytorch_rl.wrappers import NormalizedActions
import gym


class Actor(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.bn0 = nn.BatchNorm1d(num_inputs)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        x = inputs
        x = self.bn0(x)
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))

        mu = F.tanh(self.mu(x))
        return mu


class Critic(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]
        self.bn0 = nn.BatchNorm1d(num_inputs)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)

        self.linear_action = nn.Linear(num_outputs, hidden_size)
        self.bn_a = nn.BatchNorm1d(hidden_size)
        self.bn_a.weight.data.fill_(1)
        self.bn_a.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        x = inputs
        x = self.bn0(x)
        x = F.tanh(self.linear1(x))
        a = F.tanh(self.linear_action(actions))
        x = torch.cat((x, a), 1)
        x = F.tanh(self.linear2(x))

        V = self.V(x)
        return V


class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, exploration=None):
        self.actor.eval()
        mu = self.actor((Variable(state, volatile=True)).unsqueeze(0))
        self.actor.train()
        mu = mu.data
        if exploration is not None:
            mu += torch.Tensor(exploration.noise())

        return mu.clamp(-1, 1)

    def update_parameters(self, batch):
        state_batch = Variable(torch.stack(batch.state, 0))
        next_state_batch = Variable(torch.stack(batch.next_state, 0), requires_grad=False)
        action_batch = Variable(torch.stack(batch.action, 0)[:, 0])
        reward_batch = Variable(torch.cat(batch.reward))
        # mask_batch = Variable(torch.cat(batch.mask))

        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = torch.unsqueeze(reward_batch, 1)
        expected_state_action_batch = reward_batch + (self.gamma * next_state_action_values)
        expected_state_action_batch.detach_()
        self.critic_optim.zero_grad()

        state_action_batch = self.critic(state_batch, action_batch)

        value_loss = nn.MSELoss()(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic(state_batch, self.actor(state_batch))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)


if __name__ == '__main__':

    env = gym.make('Pendulum-v0')
    config = DEFAULT_CONFIG.copy()

    memory = ReplayMemory(10000000)
    ounoise = OUNoise(env.action_space.shape[0])

    env = NormalizedActions(env)
    env.seed(4)
    torch.manual_seed(4)
    np.random.seed(4)

    ddpg = DDPG(config['gamma'], config['tau'], 128,
                env.observation_space.shape[0],
                env.action_space)

    noise_scale = config['noise_scale']
    exploration_end = config['exploration_end']
    final_noise_scale = config['final_noise_scale']
    num_episodes = 1000

    for i_episode in range(num_episodes):
        obs = torch.FloatTensor(env.reset())
        ounoise.scale = (noise_scale - final_noise_scale) * max(0, exploration_end -
                                                            i_episode) / exploration_end + final_noise_scale
        ounoise.reset()
        episode_rewards = 0
        for t in range(1000):
            # Select and perform an action
            action = ddpg.select_action(obs, ounoise)
            env.render()
            next_obs, reward, done, _ = env.step(action.numpy()[0])

            next_obs = torch.FloatTensor(next_obs)
            episode_rewards += reward
            reward = torch.FloatTensor([reward])
            # Store the transition in memory
            memory.push(obs, action, next_obs, done, reward)

            # Move to the next state
            obs = next_obs

            # Perform one step of the optimization
            if len(memory) > config['batch_size'] * 5:
                for _ in range(5):
                    batch = memory.sample(config['batch_size'])
                    ddpg.update_parameters(batch)

            if done:
                break

        if i_episode % 5 == 0:
            print('i_episode: {}, reward: {}'.format(i_episode, episode_rewards))