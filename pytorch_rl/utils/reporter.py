"""Reporter reports results"""
from tensorboardX import SummaryWriter


class Reporter(object):
    def __init__(self, logdir):
        self.logdir = logdir
        self._writer = SummaryWriter(logdir)

    def report(self, step, stats):
        for k, v in stats.items():
            self._writer.add_scalar(k, v, global_step=step)