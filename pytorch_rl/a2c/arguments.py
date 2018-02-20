import torch

DEFAULT_CONFIG = {
    # RMSprop optimizer lr
    'lr': 7e-4,
    # RMSprop optimizer eps
    'eps': 1e-5,
    # RMSprop optimizer apha (default: 0.99)
    'alpha': 0.99,
    # discount factor for rewards (default: 0.99)
    'gamma': 0.99,
    # use generalized advantage estimation
    'use_gae': False,
    # gae parameter
    'tau': 0.95,
    # entropy term coefficient
    'entropy_coef': 0.01,
    # value loss coefficient
    'value_loss_coef': 0.5,
    # max norm of gradients
    'max_grad_norm': 0.5,
    # random seed
    'seed': 1,
    # how many training CPU processes to use
    'num_processes': 4,
    # number of forward steps in A2C
    'num_steps': 5,
    # number of ppo epochs
    'ppo_epoch': 4,
    # number of batches for ppo
    'num_mini_batch': 32,
    # ppo clip parameter
    'clip_param': 0.2,
    # number of frames to stack (default: 4)
    'num_stack': 4,
    # log interval, one log per n updates
    'log_interval': 10,
    # save interval, one save per n updates
    'save_interval': 100,
    # vis interval, one log per n updates
    'vis_interval': 100,
    # number of frames to train
    'num_frames': 10e6,
    # environment to train on
    'env_name': 'PongNoFrameskip-v4',
    # directory to save agent logs
    'log_dir': '/tmp/gym/',
    # directory to save agent logs
    'save_dir': '../trained_models/',
    # disables CUDA training
    'no_cuda': False,
    # use a recurrent policy
    'recurrent_policy': False,
    # disables visdom visualization
    'no_vis': False,
}


def get_args():
    from attrdict import AttrDict
    args = AttrDict(DEFAULT_CONFIG)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    return args
