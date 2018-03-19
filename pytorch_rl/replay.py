from collections import namedtuple
import random
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'done', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        state, action, next_state, done, reward = zip(*random.sample(self.memory, batch_size))

        state_batch = torch.stack(state)
        action_batch = torch.stack(action)
        next_state_batch = torch.stack(next_state)
        done_batch = torch.ByteTensor(done)
        reward_batch = torch.FloatTensor(reward)

        return Transition(state_batch, action_batch,
                          next_state_batch, done_batch,
                          reward_batch)

    def __len__(self):
        return len(self.memory)
