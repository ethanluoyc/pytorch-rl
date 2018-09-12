import os
import sys

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from pytorch_rl.a2c.agent import A2C

sys.path.append(os.path.dirname(__file__))
from pytorch_rl.a2c.envs import make_env

import torch
from pytorch_rl.a2c.agent import DEFAULT_CONFIG


def get_args():
    from attrdict import AttrDict
    args = AttrDict(DEFAULT_CONFIG)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    return args


def build_env(args):
    envs = [make_env(args.env_name, args.seed, i, args.log_dir)
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
