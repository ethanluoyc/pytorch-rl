import copy
import glob
import os
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from gym import spaces
from torch.autograd import Variable
from pytorch_rl.a2c.agent import A2C

sys.path.append(os.path.dirname(__file__))

from pytorch_rl.a2c.arguments import get_args
from pytorch_rl.a2c.envs import make_env


def build_env(args):
    envs = [make_env('Pendulum-v0', args.seed, i, args.log_dir)
            for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    return envs


def main():
    print("#######")
    print(
        "WARNING: All rewards are clipped or normalized so you need to use a monitor (see env.py) or visdom plot to get true rewards")
    print("#######")

    args = get_args()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    os.environ['OMP_NUM_THREADS'] = '1'
    env = build_env(args)
    agent = A2C(env, args)
    agent.run()


if __name__ == "__main__":
    main()
