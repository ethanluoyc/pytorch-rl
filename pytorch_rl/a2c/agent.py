import copy
import glob
import os
import time
import sys
import tqdm
import torch.multiprocessing as mp
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from pytorch_rl.a2c.envs import make_env

sys.path.append(os.path.dirname(__file__))
from pytorch_rl.a2c.model import CNNPolicy, MLPPolicy
from pytorch_rl.a2c.storage import RolloutStorage
from pytorch_rl.reporter import Reporter
from pytorch_rl.a2c.evaluator import FnEvaluator

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces


def build_model(envs, args):
    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    if len(envs.observation_space.shape) == 3:
        actor_critic = CNNPolicy(obs_shape[0], envs.action_space, args.recurrent_policy)
    else:
        assert not args.recurrent_policy, \
            "Recurrent policy is not implemented for the MLP controller"
        actor_critic = MLPPolicy(obs_shape[0], envs.action_space)

    if args.cuda:
        actor_critic.cuda()

    return actor_critic


def _evaluate(self, parameters, args):
    import warnings
    warnings.simplefilter("ignore")
    env = make_env(args.env_name, args.seed, 0, None)
    env = DummyVecEnv([env])
    env.reset()

    actor_critic = build_model(env, args)
    actor_critic.load_state_dict(parameters)

    obs_shape = env.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
    current_obs = torch.zeros(1, *obs_shape)
    states = torch.zeros(1, actor_critic.state_size)
    masks = torch.zeros(1, 1)

    def update_current_obs(obs):
        shape_dim0 = env.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = env.reset()
    update_current_obs(obs)

    episode_reward = 0
    while True:
        with torch.no_grad():
            value, action, _, states = actor_critic.act(current_obs,
                                                        states,
                                                        masks,
                                                        deterministic=True)
        states = states.data
        cpu_actions = np.atleast_2d(action.data.squeeze(1).cpu().numpy())
        # Obser reward and next obs
        obs, reward, done, _ = env.step(cpu_actions)

        # env.envs[0].render()

        episode_reward += reward
        masks.fill_(0.0 if done else 1.0)

        if current_obs.dim() == 4:
            current_obs *= masks.unsqueeze(2).unsqueeze(2)
        else:
            current_obs *= masks
        update_current_obs(obs)

        if done:
            break
    print('episode_reward = %d' % episode_reward)


class A2C(object):
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.actor_critic = build_model(env, args)

        obs_shape = env.observation_space.shape
        obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
        self.current_obs = torch.zeros(args.num_processes, *obs_shape)

    def run(self):
        self._train(self.env, self.args)

    def load_params(self, params):
        self.actor_critic.load_state_dict(params)

    def _update_current_obs(self, obs):
        shape_dim0 = self.env.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if self.args.num_stack > 1:
            self.current_obs[:, :-shape_dim0] = self.current_obs[:, shape_dim0:]
        self.current_obs[:, -shape_dim0:] = obs

    def observe(self, obs):
        self._update_current_obs(obs)

    def act(self):
        pass

    def _train(self, env, args):
        num_updates = int(args.num_frames) // args.num_steps // args.num_processes

        reporter = Reporter('./trained_models')
        evaluator = FnEvaluator(_evaluate, args)
        evaluator.start()

        obs_shape = env.observation_space.shape
        obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

        actor_critic = self.actor_critic

        if isinstance(env.action_space, spaces.Discrete):
            action_shape = 1
        else:
            action_shape = env.action_space.shape[0]

        optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
        rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, env.action_space,
                                  actor_critic.state_size)

        obs = env.reset()
        self.observe(obs)

        rollouts.observations[0].copy_(self.current_obs)

        # These variables are used to compute average rewards for all processes.
        episode_rewards = torch.zeros([args.num_processes, 1])
        final_rewards = torch.zeros([args.num_processes, 1])

        if args.cuda:
            rollouts.cuda()

        start = time.time()
        itr = tqdm.trange(num_updates)
        for j in itr:
            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, states = actor_critic.act(
                        rollouts.observations[step],
                        rollouts.states[step],
                        rollouts.masks[step])
                cpu_actions = np.reshape(action.squeeze(1).cpu().numpy(), (-1, 1))

                # Obser reward and next obs
                obs, reward, done, info = env.step(cpu_actions)
                reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
                episode_rewards += reward

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

                if args.cuda:
                    masks = masks.cuda()

                if self.current_obs.dim() == 4:
                    self.current_obs *= masks.unsqueeze(2).unsqueeze(2)
                else:
                    self.current_obs *= masks

                self.observe(obs)
                rollouts.insert(step, self.current_obs,
                                states.data, action, action_log_prob, value, reward,
                                masks)

            with torch.no_grad():
                next_value = actor_critic(rollouts.observations[-1],
                                          rollouts.states[-1],
                                          rollouts.masks[-1])[0]

                rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

            values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(
                rollouts.observations[:-1].view(-1, *obs_shape),
                rollouts.states[0].view(-1, actor_critic.state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, action_shape))

            values = values.view(args.num_steps, args.num_processes, 1)
            action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

            advantages = rollouts.returns[:-1] - values
            value_loss = advantages.pow(2).mean()

            # we do not backprop from advantages
            action_loss = -(advantages.detach() * action_log_probs).mean()

            optimizer.zero_grad()
            loss = value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef
            loss.backward()
            nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()

            rollouts.after_update()

            if j % args.save_interval == 0 and args.save_dir != "":
                save_path = os.path.join(args.save_dir, 'a2c')
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                # A really ugly way to save a model to CPU
                save_model = actor_critic
                if args.cuda:
                    save_model = copy.deepcopy(actor_critic).cpu()

                torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

            if j % args.log_interval == 0:
                end = time.time()
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                # itr.set_postfix_str(
                #     "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                #         format(j, total_num_steps,
                #                int(total_num_steps / (end - start)),
                #                final_rewards.mean(),
                #                final_rewards.median(),
                #                final_rewards.min(),
                #                final_rewards.max(), dist_entropy.data[0],
                #                value_loss.data[0], action_loss.data[0]))

                itr.set_postfix_str(
                    "Updates {}, num timesteps {}, FPS {}, mean {:.1f}".
                        format(j, total_num_steps,
                               int(total_num_steps / (end - start)),
                               final_rewards.mean()))

                reporter.report(j, dict(value_loss=value_loss.item(),
                                        dist_entropy=dist_entropy.item(),
                                        final_rewards=final_rewards.mean()))