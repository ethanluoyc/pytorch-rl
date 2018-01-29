"""Advantageous Actor Critic (A2C)"""
from torch import nn
import torch.functional as F


# TODO Gaussian policy (square error)
# TODO Discrete policy (cross entropy)


class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
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