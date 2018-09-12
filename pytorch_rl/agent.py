

class Agent(object):
    def __init__(self):
        pass

    def episode_begin(self, obs):
        pass

    def step(self, obs, reward, done):
        pass

    def episode_end(self):
        pass

    def get_state(self):
        return {}

    def load(self, _checkpoint_dir,
             latest_checkpoint_version,
             experiment_data):
        return True


def make_agent(env):
    pass