from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.autograd import Variable
from pytorch_rl.replay import ReplayMemory
from pytorch_rl.ounoise import OUNoise
from pytorch_rl.wrappers import NormalizedActions


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Policy(nn.Module):
    def __init__(self, dim_in, dim_act):
        super(Policy, self).__init__()

        hidden_size = 128
        self.bn0 = nn.BatchNorm1d(dim_in)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(dim_in, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        self.mu = nn.Linear(hidden_size, dim_act)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

        self.L = nn.Linear(hidden_size, dim_act ** 2)
        self.L.weight.data.mul_(0.1)
        self.L.bias.data.mul_(0.1)

        self.tril_mask = Variable(torch.tril(torch.ones(
            dim_act, dim_act), diagonal=-1).unsqueeze(0))
        self.diag_mask = Variable(torch.diag(torch.diag(
            torch.ones(dim_act, dim_act))).unsqueeze(0))

    def forward(self, inputs):
        x, u = inputs
        x = self.bn0(x)
        x = F.leaky_relu(self.bn1(self.linear1(x)))
        x = F.leaky_relu(self.bn2(self.linear2(x)))

        V = self.V(x)
        mu = F.leaky_relu(self.mu(x))

        Q = None
        if u is not None:
            num_outputs = mu.size(1)
            L = self.L(x).view(-1, num_outputs, num_outputs)
            L = L * self.tril_mask.expand_as(L) + torch.exp(L) * self.diag_mask.expand_as(L)
            P = torch.bmm(L, L.transpose(2, 1))

            u_mu = (u - mu).unsqueeze(2)
            A = -0.5 * torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]

            Q = A + V

        return mu, Q, V


DEFAULT_CONFIG = {
    'batch_size': 128,
    'gamma': 0.99,
    'eps_start': 0.9,
    'eps_end': 0.05,
    'eps_decay': 200,
    'clip_grad': 1,
    'tau': 0.001,
    'noise_scale': 0.3,
    'exploration_end': 100,
    'final_noise_scale': 0.3,
}


class NAF(object):
    def __init__(self):
        self.model = Policy(env.observation_space.shape[0], env.action_space.shape[0])
        self.target_model = Policy(env.observation_space.shape[0], env.action_space.shape[0])
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def select_action(self, state, noise):
        self.model.eval()
        action, _, _ = self.model((Variable(state, requires_grad=False).unsqueeze(0), None))
        action = action[0]
        action += Variable(torch.from_numpy(noise.noise()), requires_grad=False).float()
        self.model.train()
        return torch.clamp(action, -1, 1).data

    def update_parameters(self, batch):
        self.model.train()
        state_batch = Variable(torch.stack(batch.state, 0))
        next_state_batch = Variable(torch.stack(batch.next_state, 0), requires_grad=False)
        action_batch = Variable(torch.stack(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(torch.FloatTensor([[1.0] if not d else [0.0] for d in batch.done])))

        _, _, next_state_values = self.target_model((next_state_batch, None))

        reward_batch = (torch.unsqueeze(reward_batch, 1))

        # expected_state_action_values = reward_batch + (next_state_values * self.gamma)
        expected_state_action_values = reward_batch + (next_state_values * config['gamma']) * mask_batch.view(-1, 1)

        expected_state_action_values = expected_state_action_values.detach()

        _, state_action_values, _ = self.model((state_batch, action_batch))

        loss = nn.MSELoss()(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
        self.optimizer.step()

        soft_update(self.target_model, self.model, config['tau'])


if __name__ == '__main__':
    # env = gym.make('Pendulum-v0')
    from pixel2torque.envs.reacher_my import ReacherBulletEnv
    from gym.wrappers import TimeLimit, Monitor

    env = ReacherBulletEnv()
    env.spec = None
    env = TimeLimit(env, max_episode_steps=200)

    experiment_dir = 'experiments/naf-ReacherFixedTarget'
    env = Monitor(env, experiment_dir, force=True)

    config = DEFAULT_CONFIG.copy()

    memory = ReplayMemory(10000000)
    ounoise = OUNoise(env.action_space.shape[0])

    env = NormalizedActions(env)
    env.seed(4)
    torch.manual_seed(4)
    np.random.seed(4)

    naf = NAF()

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
            action = naf.select_action(obs, noise=ounoise)
            # env.render()
            next_obs, reward, done, _ = env.step(action.numpy())

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
                    naf.update_parameters(batch)

            if done:
                break

        if i_episode % 20 == 0:
            print('i_episode: {}, reward: {}'.format(i_episode, episode_rewards))
            torch.save(naf.model.state_dict(), 'model.pth')
