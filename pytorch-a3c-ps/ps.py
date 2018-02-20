import ray


@ray.remote
class AsyncParameterServer(object):
    def __init__(self, keys, values):
        values = [value.copy() for value in values]
        self.weights = dict(zip(keys, values))

    def push(self, keys, values):
        for key, value in zip(keys, values):
            self.weights[key] += value

    def pull(self, keys):
        return [self.weights[key] for key in keys]


@ray.remote
class SyncParameterServer(object):
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def apply_gradients(self, *gradients):
        pass # TODO

    def get_weights(self):
        pass # TODO

if __name__ == '__main__':
    pass