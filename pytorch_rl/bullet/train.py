import gym
from pytorch_rl.agent import Agent
from pytorch_rl.bullet.runner import Runner
import time
import logging
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os

def build_agent(env):
    class RandomAgent(Agent):
        def episode_begin(self, obs):
            return env.action_space.sample()

        def step(self, obs, reward, done):
            time.sleep(0.001)
            return env.action_space.sample()

        def episode_end(self):
            pass

    return RandomAgent()


def main():
    logging.basicConfig(level=logging.INFO)
    env_name = 'Pendulum-v0'
    logdir = './tmp'

    # TODO refactor this
    def on_end_episode(runner, t):
        agent = runner._agent
        env = runner._env

        frames = []
        obs = env.reset()
        frames.append(obs)

        writer = imageio.get_writer(os.path.join(
            runner._logdir, '{}.mp4'.format(t)),
            fps=30
        )

        action = agent.episode_begin(obs)
        while True:
            obs, reward, done, _ = env.step(action)
            writer.append_data(env.render('rgb_array'))
            action = agent.step(obs, reward, done)
            if done:
                agent.episode_end()
                break
        writer.close()

    runner = Runner(env_name, logdir,
                    build_agent,
                    lambda env_name: gym.make(env_name),
                    num_iterations=10,
                    training_steps=1000,
                    evaluation_steps=1000)

    runner.add_hook('after_iteration', on_end_episode)
    runner.run()


if __name__ == '__main__':
    main()