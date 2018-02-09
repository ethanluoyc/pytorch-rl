"""Advantageous Actor Critic (A2C)"""
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
import numpy as np


# TODO Gaussian policy (square error)
# TODO Discrete policy (cross entropy)


class PG(nn.Module):
    def __init__(self):
        super(PG, self).__init__()
        self.fc1 = nn.Linear(4, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.head = nn.Linear(200, 2)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        return self.head(x)

    def train_step(self, actions, states, q_values, optimizer):
        # Given:
        # actions -  (N * T) x Da tensor of actions
        # states -   (N * T) x Ds tensor of states
        # q_values - (N * T) x Ds tensor of q_values

        predicted_actions = self(states)

        nll = nn.CrossEntropyLoss(reduce=False)(predicted_actions, actions)
        l = nll.mul(q_values).mean()
        l.backward()

        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)

        optimizer.step()


class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
        self.v_fc1 = nn.Linear(4, 200)
        self.v_bn1 = nn.BatchNorm1d(200)
        self.v_fc2 = nn.Linear(200, 200)
        self.v_bn2 = nn.BatchNorm1d(200)
        self.v_head = nn.Linear(200, 1)

        self.p_fc1 = nn.Linear(4, 200)
        self.p_bn1 = nn.BatchNorm1d(200)
        self.p_fc2 = nn.Linear(200, 200)
        self.p_bn2 = nn.BatchNorm1d(200)
        self.p_head = nn.Linear(200, 2)

    def _value(self, s):
        x = F.leaky_relu((self.v_fc1(s)))
        x = F.leaky_relu((self.v_fc2(x)))
        return self.v_head(x)

    def _policy(self, s):
        x = F.leaky_relu((self.p_fc1(s)))
        x = F.leaky_relu((self.p_fc2(x)))
        return self.p_head(x)

    def _var(self, x):
        return torch.autograd.Variable(torch.from_numpy(x)).float().unsqueeze(0)

    def train_me(self, env, gamma=.99):
        optimizer = torch.optim.Adam(self.parameters())

        for _ in range(1000):
            s = env.reset()
            s = self._var(s)
            while True:
                predicted_actions = self._policy(s)

                actions = np.atleast_1d(env.action_space.sample())
                actions = self._var(actions).long()

                snext, r, done, _ = env.step(predicted_actions.max(1)[1].numpy()[0])

                snext = self._var(snext)
                r = self._var(np.atleast_1d(r))

                Vsnext = self._value(snext)
                Vs = self._value(s)

                target = r + gamma * Vsnext
                target = target.detach()

                # print(Vs); print(target)

                loss = F.smooth_l1_loss(Vs, target)

                # optimizer.zero_grad()
                # loss.backward()
                # for param in self.parameters():
                #     if param.grad is not None:
                #         param.grad.data.clamp_(-1, 1)
                # optimizer.step()

                A = target - Vs  # Advantage
                # print(predicted_actions)
                # print(actions)

                nll = nn.CrossEntropyLoss(reduce=False)(predicted_actions, actions[:, 0])
                l = nll.mul(A).mean()

                optimizer.zero_grad()
                l.backward()
                for param in self.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-1, 1)
                optimizer.step()

                s = snext
                if done:
                    break


if __name__ == "__main__":
    import gym

    env = gym.make('CartPole-v0')
    learner = A2C()
    learner.train_me(env)
