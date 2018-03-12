import torch.multiprocessing as mp
import gym
import numpy as np
import logging

_logger = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(self, agent_fn, env_fn, opt={}):
        self.agent_fn = agent_fn
        self.env_fn = env_fn
        self.opt = opt
        self._q = mp.Queue()

        self._worker = mp.Process(target=self._worker)

    def start(self):
        self._worker.start()

    def push(self, params):
        self._q.put(params)

    def _worker(self):
        agent = self.agent_fn()
        env = self.env_fn()

        stop = False
        import signal

        def _handle_sigterm(*args, **kwargs):
            nonlocal stop
            stop = True
            _logger.info('shutting down')

        signal.signal(signal.SIGTERM, _handle_sigterm)

        while True:
            if stop:
                env.close()
                break
            parameters = self._q.get()
            _logger.info('Get a new snapshot, evaluating it')
            agent.load_params(parameters)
            obs = env.reset()
            while True:
                agent.observe(obs)
                action = agent.act()
                obs, reward, done, info = env.step(action)
                env.render()
                if done:
                    break


class _MockAgent(object):
    def __init__(self, ):
        self.observation = None

    def load_params(self, params):
        pass

    def observe(self, observation):
        self.observation = observation

    def act(self):
        return np.random.randn(1,)

    def reset(self):
        self.observation = None


if __name__ == '__main__':
    agent_fn = lambda : _MockAgent()
    env_fn = lambda : gym.make('Pendulum-v0')
    evaluator = Evaluator(agent_fn, env_fn)
    evaluator.start()
    for _ in range(10):
        evaluator.push({})

    import time
    time.sleep(5)
    evaluator._worker.terminate()
