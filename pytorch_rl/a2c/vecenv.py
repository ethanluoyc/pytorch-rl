from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import ray
from baselines.common.vec_env import VecEnvWrapper

import gym

_initialized = False
# ray.init(redirect_output=True)


@ray.remote
class RayEnv(object):
    def __init__(self, fn):
        # Tell numpy to only use one core. If we don't do this, each actor may
        # try to use all of the cores and the resulting contention may result
        # in no speedup over the serial version. Note that if numpy is using
        # OpenBLAS, then you need to set OPENBLAS_NUM_THREADS=1, and you
        # probably need to do it from the command line (so it happens before
        # numpy is imported).
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = fn()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def _call(self, name, args, kwargs):
        return getattr(self.env, name)(*args, **kwargs)

    def action_space(self):
        return self.env.action_space

    def observation_space(self):
        return self.env.observation_space


class VecEnv(gym.Env):
    def __init__(self, env_create_fn, num_envs=4):
        self._envs = [RayEnv.remote(env_create_fn) for _ in range(num_envs)]

    def _reset(self):
        return np.stack(ray.get([e.reset.remote() for e in self._envs]))

    def _step(self, actions):
        results = ray.get([e.step.remote(a) for e, a in zip(self._envs, actions)])
        obs_next, rewards, done, info = zip(*results)
        return np.stack(obs_next), np.array(rewards), np.array(done), list(info)

    @property
    def action_space(self):
        return ray.get(self._envs[0].action_space.remote())

    @property
    def observation_space(self):
        return ray.get(self._envs[0].observation_space.remote())


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class VecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs


    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)


if __name__ == '__main__':
    envs = VecEnv()
    print(envs.reset())
    actions = np.array([envs.action_space.sample() for _ in envs._envs])
    next_obs, rewards, done, info = envs.step(actions)

    print(next_obs)
    print(rewards)
    print(done)
    print(info)