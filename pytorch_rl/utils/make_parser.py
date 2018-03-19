import argparse

config = {
    'action_space': 2,
    'observation_space': 'foo',
    'optim': {'batch_size': 1,
              'lr': 2,
              'more_sub': {
                  'a': 'a',
                  'b': 'b'}
              }
}


def make_parser(config):
    parser = argparse.ArgumentParser()

    def _add_subopt(opt, prefix=''):
        for key, val in opt.items():
            if isinstance(val, dict):
                if len(prefix) == 0:
                    _add_subopt(val, prefix + key)
                else:
                    _add_subopt(val, prefix + '_' + key)
            else:
                if len(prefix) == 0:
                    parser.add_argument('--' + '_'.join([prefix, key]), default=val)
                else:
                    parser.add_argument('--' + key, default=val)

    _add_subopt(config)
    parser.print_help()


make_parser(config)
