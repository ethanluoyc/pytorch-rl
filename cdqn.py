import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import random
import torch
import torch.optim as optim
from torch.autograd import Variable
from replay import Transition, ReplayMemory
import gym
import math
import itertools


env = gym.make('Pendulum-v0')
use_cuda = False

# TODO exponentiate the diagonal
class NAF(nn.Module):
    def __init__(self, dim_in, dim_act):
        super(NAF, self).__init__()
        self.fc1 = nn.Linear(dim_in, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc_V = nn.Linear(64, 1)
        self.fc_mu= nn.Linear(64, dim_act)
        self.fc_L = nn.Linear(64, (dim_act)*dim_act)

        self.relu = nn.LeakyReLU()

    def forward(self, obs, u):
        return self.q_value(obs)(u)

    def value(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        return self.fc_V(x)

    def q_value(self, obs):
        x = self.relu(self.bn1(self.fc1(obs)))

        V = self.fc_V(x)[:, 0]
        L = self.fc_L(x).view(-1, 1, 1)
        mean = self.fc_mu(x)
        P = torch.bmm(L, L)

        def fn(u):
            umean = (u - mean)  # (u - \mu)
            A = -.5 * (umean * ((P @ umean.unsqueeze(2)).squeeze(2))).sum(dim=1)
            Q = A + V
            return Q
        return fn

    def q_max(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        mean = self.fc_mu(x)
        return mean


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

model = NAF(env.observation_space.shape[0], env.action_space.shape[0])

if use_cuda:
    model.cuda()

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    model.eval()
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model.q_max(
            Variable(state.unsqueeze(0), volatile=True).float()).data.view(1, 1)
    else:
        return torch.from_numpy(np.array([env.action_space.sample()]))

episode_durations = []


def optimize_model():
    model.train()
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.stack([s for s in batch.next_state
                                                  if s is not None], 0),
                                     volatile=True)

    state_batch = Variable(torch.stack(batch.state, 0))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    # print(state_batch.size())
    state_action_values = model.q_value(state_batch)(action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(state_batch.size()[0]).type(torch.FloatTensor))
    next_state_values[non_final_mask] = model.q_max(non_final_next_states)

    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    # print(state_action_values.size(), expected_state_action_values.size())
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

env.seed(42)
env.reset()


num_episodes = 1000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    obs = torch.FloatTensor(env.reset())
    state = obs
    episode_rewards = 0
    for t in itertools.count():
        # Select and perform an action
        action = select_action(state)
        env.render()
        obs, reward, done, _ = env.step(action[0].numpy())
        obs = torch.FloatTensor(obs)
        episode_rewards += reward
        reward = torch.FloatTensor([reward])

        # Observe new state
        if not done:
            next_state = obs
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break

    if i_episode % 100 == 0:
        print('i_episode: {}, reward: {}'.format(i_episode, episode_rewards))

torch.save(model.parameters(), 'model.pth')