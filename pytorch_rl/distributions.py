import torch
from torch.distributions import constraints, Distribution
import math


class DiagonalNormal(Distribution):
    arg_constraints = {'loc': constraints.real_vector,
                       'var': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, variance=None, validate_args=None):
        event_shape = torch.Size(loc.shape[-1:])
        batch_shape = loc.shape[:-1]
        self.loc = loc
        self._variance = variance
        super(DiagonalNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self._variance

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(*shape).normal_()
        return self.loc + eps * self.variance

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        log_probs = -0.5 * (diff / self.stddev).pow(2) - 0.5 * math.log(2 * math.pi) - self.stddev.log()
        log_probs = log_probs.sum(-1)
        return log_probs

    def entropy(self):
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + self.stddev.log()
        entropy = entropy.sum(-1)
        return entropy
