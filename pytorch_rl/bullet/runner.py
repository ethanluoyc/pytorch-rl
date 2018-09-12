from tensorboardX import SummaryWriter
import os
import subprocess
import sys
import logging
import json
import collections

from pytorch_rl.bullet import checkpointer
from pytorch_rl.bullet import hooks

logger = logging.getLogger(__name__)

_CHECKPOINT_PREFIX = 'ckpt'


def is_under_git_control():
    return subprocess.run(['git', 'rev-parse']).returncode == 0



class Runner(object):
    def __init__(self,
                 env_name,
                 logdir,
                 create_agent_fn,
                 create_env_fn,
                 num_iterations=100,
                 training_steps=15 * 1000,
                 evaluation_steps=10 * 1000,
                 checkpoint_every_n=1,
                 log_every_n=1):

        self._env_name = env_name
        self._create_agent_fn = create_agent_fn
        self._create_env_fn = create_env_fn

        self.num_iterations = num_iterations
        self.training_steps = training_steps
        self.evaluation_steps = evaluation_steps

        self._env = self._create_env_fn(self._env_name)
        self._agent = self._create_agent_fn(self._env)

        self._logdir = logdir
        self._logger = SummaryWriter(logdir)
        self._log_every_n = log_every_n

        self.checkpoint_every_n = checkpoint_every_n
        self._checkpoint_dir = os.path.join(logdir, 'checkpoints')

        self._create_dir(logdir)
        self._save_run_info(logdir)

        self._hooks = {
            hooks.BEFORE_ITERATION: [],
            hooks.AFTER_ITERATION: [],
        }

        self.current_iteration = 0

    def _create_dir(self, logdir):
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(os.path.join(logdir, 'logs'), exist_ok=True)

    def _save_run_info(self, outdir):
        # Save all the arguments
        # with open(os.path.join(outdir, 'args.txt'), 'w') as f:
        #     if isinstance(args, argparse.Namespace):
        #         args = vars(args)
        #     f.write(json.dumps(args))

        # Save all the environment variables
        with open(os.path.join(outdir, 'environ.txt'), 'w') as f:
            f.write(json.dumps(dict(os.environ)))

        # Save the command
        with open(os.path.join(outdir, 'command.txt'), 'w') as f:
            f.write(' '.join(sys.argv))

        if is_under_git_control():
            # Save `git rev-parse HEAD` (SHA of the current commit)
            with open(os.path.join(outdir, 'git-head.txt'), 'wb') as f:
                f.write(subprocess.check_output('git rev-parse HEAD'.split()))

            # Save `git status`
            with open(os.path.join(outdir, 'git-status.txt'), 'wb') as f:
                f.write(subprocess.check_output('git status'.split()))

            # Save `git log`
            with open(os.path.join(outdir, 'git-log.txt'), 'wb') as f:
                f.write(subprocess.check_output('git log'.split()))

            # Save `git diff`
            with open(os.path.join(outdir, 'git-diff.txt'), 'wb') as f:
                f.write(subprocess.check_output('git diff'.split()))

    def add_hook(self, stage, hook):
        assert stage in self._hooks
        self._hooks[stage].append(hook)

    def _run_hooks(self, stage, iteration_number):
        for hook in self._hooks[stage]:
            hook(self, iteration_number)

    def _run_one_episode(self):
        env = self._env
        agent = self._agent
        num_steps = 0
        episode_reward = 0

        obs = env.reset()
        action = agent.episode_begin(obs)
        while True:
            obs, reward, done, _ = env.step(action)
            action = agent.step(obs, reward, done)
            episode_reward += reward
            num_steps += 1
            if done:
                agent.episode_end()
                break

        return episode_reward, num_steps

    def _run_one_phase(self, min_steps):
        sum_rewards = 0.
        num_episodes = 0
        step_count = 0
        while step_count < min_steps:
            episode_reward, episode_steps = self._run_one_episode()

            sum_rewards += episode_reward
            step_count += episode_steps
            num_episodes += 1

            sys.stdout.write('Steps executed: {} '.format(step_count) +
                             'Episode length: {} '.format(episode_steps) +
                             'Return: {}\r'.format(episode_reward))
            sys.stdout.flush()

        return sum_rewards / num_episodes, step_count

    def _run_train_phase(self):
        self._agent.eval_mode = False
        return self._run_one_phase(self.training_steps)

    def _run_eval_phase(self):
        self._agent.eval_mode = True
        return self._run_one_phase(self.evaluation_steps)

    def _log_stats(self, stats, iter_num):
        for stats_name, stats_value in stats.items():
            self._logger.add_scalar(stats_name, stats_value, iter_num)

    def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
        """Reloads the latest checkpoint if it exists.
        This method will first create a `Checkpointer` object and then call
        `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
        checkpoint in self._checkpoint_dir, and what the largest file number is.
        If a valid checkpoint file is found, it will load the bundled data from this
        file and will pass it to the agent for it to reload its data.
        If the agent is able to successfully unbundle, this method will verify that
        the unbundled data contains the keys,'logs' and 'current_iteration'. It will
        then load the `Logger`'s data from the bundle, and will return the iteration
        number keyed by 'current_iteration' as one of the return values (along with
        the `Checkpointer` object).
        Args:
          checkpoint_file_prefix: str, the checkpoint file prefix.
        Returns:
          start_iteration: int, the iteration number to start the experiment from.
          experiment_checkpointer: `Checkpointer` object for the experiment.
        """
        self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                       checkpoint_file_prefix)
        self._start_iteration = 0
        # Check if checkpoint exists. Note that the existence of checkpoint 0 means
        # that we have finished iteration 0 (so we will start from iteration 1).
        latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
            self._checkpoint_dir)
        if latest_checkpoint_version >= 0:
            experiment_data = self._checkpointer.load_checkpoint(
                latest_checkpoint_version)
            if self._agent.load(
                    self._checkpoint_dir, latest_checkpoint_version, experiment_data):
                assert 'logs' in experiment_data
                assert 'current_iteration' in experiment_data
                self._logger.data = experiment_data['logs']
                self._start_iteration = experiment_data['current_iteration'] + 1
                logger.info('Reloaded checkpoint and will start from iteration %d',
                            self._start_iteration)

    def run(self):
        self._initialize_checkpointer_and_maybe_resume(_CHECKPOINT_PREFIX)
        self.current_iteration = self._start_iteration
        if self.current_iteration >= self.num_iterations:
            logger.warning("start_iteration ({}) >= num_iterations ({})"
                           .format(self._start_iteration,
                                   self.num_iterations))
            return

        while self.current_iteration < self.num_iterations:
            logger.info('Running iteration # {}'.format(self.current_iteration))

            self._run_hooks('before_iteration', self.current_iteration)
            train_avg_rew, train_num_steps = self._run_train_phase()
            eval_avg_rew, eval_num_steps = self._run_eval_phase()
            self._run_hooks('after_iteration', self.current_iteration)

            logger.info('Train\t rewards: {}, steps: {}\n'
                        'Eval\t rewards: {}, steps: {}'.format(train_avg_rew, train_num_steps,
                                                               eval_avg_rew, eval_num_steps))

            stats = {
                'Train/AverageReward': train_avg_rew,
                'Train/NumSteps': train_num_steps,
                'Eval/AverageReward': eval_avg_rew,
                'Eval/NumSteps': eval_num_steps
            }

            if self.current_iteration % self._log_every_n == 0:
                self._log_stats(stats, self.current_iteration)

            if self.current_iteration % self.checkpoint_every_n == 0:
                self._checkpointer.save_checkpoint(
                    self.current_iteration,
                    {'agent': self._agent.get_state(),
                     'logs': [],
                     'current_iteration': self.current_iteration}
                )

            self.current_iteration += 1
