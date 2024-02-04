# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runner for evaluating using a fixed number of episodes."""

import functools
import sys
import time

from absl import logging
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment
import gin
import jax
import numpy as np
import tensorflow as tf
from .normalize_score import normalize_score
import gym


def create_environment(game_name: str):
  full_game_name = f'{game_name}NoFrameskip-v4'
  env = gym.make(full_game_name, full_action_space=True)
  env = env.env
  env = atari_lib.AtariPreprocessing(env)
  return env


def create_env_wrapper(game_name: str):
  create_env_fn = functools.partial(
      create_environment, game_name=game_name
  )
  def inner_create(*args, **kwargs):
    env = create_env_fn(*args, **kwargs)
    env.cum_length = 0
    env.cum_reward = 0
    return env

  return inner_create


@gin.configurable
class DataEfficientAtariRunner(run_experiment.Runner):
  """Runner for evaluating using a fixed number of episodes rather than steps.

  Also restricts data collection to a strict cap,
  following conventions in data-efficient RL research.
  """

  def __init__(
      self,
      base_dir,
      create_agent_fn,
  ):
    """Specify the number of evaluation episodes."""
    game_name = "Breakout"  # dummy
    create_environment_fn = functools.partial(
        create_environment, game_name=game_name
    )
    super().__init__(
        base_dir, create_agent_fn, create_environment_fn=create_environment_fn)

    self.num_steps = 0

    self.max_noops = 30
    self.parallel_eval = True
    self.num_eval_envs = 100
    self.eval_one_to_one = True

  def _run_one_phase(self,
                     envs,
                     steps,
                     max_episodes,
                     statistics,
                     run_mode_str,
                     game_name: str,
                     needs_reset=False,
                     one_to_one=False,
                     resume_state=None):
    """Runs the agent/environment loop until a desired number of steps.

    We terminate precisely when the desired number of steps has been reached,
    unlike some other implementations.

    Args:
      envs: environments to use in this phase.
      steps: int, how many steps to run in this phase (or None).
      max_episodes: int, maximum number of episodes to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.
      needs_reset: bool, whether to reset all environments before starting.
      one_to_one: bool, whether to precisely match each episode in
        `max_episodes` to an environment in `envs`. True is faster but only
        works in some situations (e.g., evaluation).
      resume_state: bool, whether to have the agent resume its prior state for
        the current mode.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the
      sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.

    (episode_lengths, episode_returns, state, envs) = self._run_parallel(
        episodes=max_episodes,
        envs=envs,
        one_to_one=one_to_one,
        needs_reset=needs_reset,
        resume_state=resume_state,
        max_steps=steps,
        game_name=game_name,
    )

    for episode_length, episode_return in zip(episode_lengths, episode_returns):
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return
      })
      if run_mode_str == 'train':
        # we use one extra frame at the starting
        self.num_steps += episode_length
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      sys.stdout.flush()
      if self._summary_writer is not None:
        with self._summary_writer.as_default():
          _ = (
              tf.summary.scalar(
                  'train_episode_returns',
                  float(episode_return),
                  step=self.num_steps,
              ),
          )
          _ = tf.summary.scalar(
              'train_episode_lengths',
              float(episode_length),
              step=self.num_steps,
          )
    return step_count, sum_returns, num_episodes, state, envs

  def _initialize_episode(self, envs):
    """Initialization for a new episode.

    Args:
      envs: Environments to initialize episodes for.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    observations = []
    for env in envs:
      initial_observation = env.reset()
      if self.max_noops > 0:
        self._agent._rng, rng = jax.random.split(
            self._agent._rng  # pylint: disable=protected-access
        )
        num_noops = jax.random.randint(rng, (), 0, self.max_noops)
        for _ in range(num_noops):
          initial_observation, _, terminal, _ = env.step(0)
          if terminal:
            initial_observation = env.reset()
      observations.append(initial_observation)
    initial_observation = np.stack(observations, 0)

    return initial_observation

  def _run_parallel(self,
                    envs,
                    game_name: str,
                    episodes=None,
                    max_steps=None,
                    one_to_one=False,
                    needs_reset=True,
                    resume_state=None):
    """Executes a full trajectory of the agent interacting with the environment.

    Args:
      envs: Environments to step in.
      episodes: Optional int, how many episodes to run. Unbounded if None.
      max_steps: Optional int, how many steps to run. Unbounded if None.
      one_to_one: Bool, whether to couple each episode to an environment.
      needs_reset: Bool, whether to reset environments before beginning.
      resume_state: State tuple to resume.

    Returns:
      The number of steps taken and the total reward.
    """
    # You can't ask for 200 episodes run one-to-one on 100 envs
    if one_to_one:
      assert episodes is None or episodes == len(envs)

    # Create envs
    live_envs = list(range(len(envs)))

    if needs_reset:
      new_obs = self._initialize_episode(envs)
      new_obses = np.zeros((2, len(envs), *self._agent.observation_shape, 1))
      self._agent.reset_all(new_obs)

      rewards = np.zeros((len(envs),))
      terminals = np.zeros((len(envs),))
      episode_end = np.zeros((len(envs),))

      cum_rewards = []
      cum_lengths = []
    else:
      assert resume_state is not None
      (new_obses, rewards, terminals, episode_end, cum_rewards, cum_lengths) = (
          resume_state
      )

    total_steps = 0
    total_episodes = 0
    max_steps = np.inf if max_steps is None else max_steps
    step = 0

    # Keep interacting until we reach a terminal state.
    while True:
      b = 0
      step += 1
      episode_end.fill(0)
      total_steps += len(live_envs)
      actions = self._agent.step()

      # The agent may be hanging on to the previous new_obs, so we don't
      # want to change it yet.
      # By alternating, we can make sure we don't end up logging
      # with an offset.
      new_obs = new_obses[step % 2]

      # don't want to do a for-loop since live envs may change
      while b < len(live_envs):
        env_id = live_envs[b]
        obs, reward, d, _ = envs[env_id].step(actions[b])
        envs[env_id].cum_length += 1
        envs[env_id].cum_reward += reward
        new_obs[b] = obs
        rewards[b] = reward
        terminals[b] = d

        if (envs[env_id].game_over or
            envs[env_id].cum_length == self._max_steps_per_episode):
          total_episodes += 1
          cum_rewards.append(envs[env_id].cum_reward)
          cum_lengths.append(envs[env_id].cum_length)
          envs[env_id].cum_length = 0
          envs[env_id].cum_reward = 0

          human_norm_ret = normalize_score(cum_rewards[-1], game_name)

          print(f'Steps executed: {total_steps} ' +
                f'Num episodes: {len(cum_rewards)} ' +
                f'Episode length: {cum_lengths[-1]} ' +
                f'Return: {cum_rewards[-1]:4.1f} ' +
                f'Normalized Return: {np.round(human_norm_ret, 3)}')
          self._maybe_save_single_summary(self.num_steps + total_steps,
                                          cum_rewards[-1], cum_lengths[-1], game_name)

          if one_to_one:
            new_obses = delete_ind_from_array(new_obses, b, axis=1)
            new_obs = new_obses[step % 2]
            actions = delete_ind_from_array(actions, b)
            rewards = delete_ind_from_array(rewards, b)
            terminals = delete_ind_from_array(terminals, b)
            self._agent.delete_one(b)
            del live_envs[b]
            b -= 1  # live_envs[b] is now the next env, so go back one.
          else:
            episode_end[b] = 1
            new_obs[b] = self._initialize_episode([envs[env_id]])
            self._agent.reset_one(env_id=b)
        elif d:
          self._agent.reset_one(env_id=b)

        b += 1

      if self._clip_rewards:
        # Perform reward clipping.
        rewards = np.clip(rewards, -1, 1)

      self._agent.log_transition(new_obs, actions, rewards, terminals,
                                 episode_end)

      if (
          not live_envs
          or (max_steps is not None and total_steps > max_steps)
          or (episodes is not None and total_episodes > episodes)
      ):
        break

    state = (new_obses, rewards, terminals, episode_end, cum_rewards,
             cum_lengths)
    return cum_lengths, cum_rewards, state, envs

  def _run_train_phase(self, game_name: str, iteration: int, statistics):
    """Run training phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
      average_steps_per_second: float, The average number of steps per
      second.
    """
    # Perform the training phase, during which the agent learns.
    self._agent.eval_mode = False
    start_time = time.time()
    create_env = create_env_wrapper(game_name)
    train_envs = [create_env()]  # means num_train_envs=1

    (
        number_steps,
        sum_returns,
        num_episodes,
        _,
        _,
    ) = self._run_one_phase(
        train_envs,
        self._training_steps,
        max_episodes=None,
        statistics=statistics,
        run_mode_str='train',
        needs_reset=True,
        resume_state=None,
        game_name=game_name,
    )
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    statistics.append({'train_average_return': average_return})
    human_norm_ret = normalize_score(average_return, game_name)
    statistics.append({'train_average_normalized_score': human_norm_ret})
    time_delta = time.time() - start_time
    average_steps_per_second = number_steps / time_delta
    statistics.append(
        {'train_average_steps_per_second': average_steps_per_second}
    )
    logging.info(
        'Average undiscounted return per training episode: %.2f', average_return
    )
    logging.info(
        'Average normalized return per training episode: %.2f', human_norm_ret
    )
    logging.info(
        'Average training steps per second: %.2f', average_steps_per_second
    )
    self._agent.cache_train_state()

    with self._summary_writer.as_default():
      tf.summary.scalar('Train/NumEpisodes', num_episodes, step=iteration)
      tf.summary.scalar('Train/AverageReturns', average_return, step=iteration)
      tf.summary.scalar('Train/AverageNormalizedScore', human_norm_ret, step=iteration)
      tf.summary.scalar('Train/AverageStepsPerSecond', average_steps_per_second, step=iteration)

  def _run_eval_phase(self, game_name: str, iteration: int, statistics):
    """Run evaluation phase.

    Args:
        statistics: `IterationStatistics` object which records the experimental
          results. Note - This object is modified by this method.

    Returns:
        num_episodes: int, The number of episodes run in this phase.
        average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    create_env = create_env_wrapper(game_name)
    eval_envs = [
        create_env() for _ in range(self.num_eval_envs)
    ]
    _, sum_returns, num_episodes, _, _ = self._run_one_phase(
        eval_envs,
        steps=None,
        max_episodes=self.num_eval_envs,
        statistics=statistics,
        needs_reset=True,
        resume_state=None,
        one_to_one=self.eval_one_to_one,
        run_mode_str='eval',
        game_name=game_name,
    )
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    logging.info(
        'Average undiscounted return per evaluation episode: %.2f',
        average_return,
    )
    statistics.append({'eval_average_return': average_return})
    human_norm_return = normalize_score(average_return, game_name)
    statistics.append({'train_average_normalized_score': human_norm_return})
    logging.info(
        'Average normalized return per evaluation episode: %.2f',
        human_norm_return,
    )

    with self._summary_writer.as_default():
      tf.summary.scalar(f'Eval/{game_name}/NumEpisodes', num_episodes, step=iteration)
      tf.summary.scalar(f'Eval/{game_name}/AverageReturns', average_return, step=iteration)
      tf.summary.scalar(f'Eval/{game_name}/NormalizedScore', human_norm_return, step=iteration)

  def _maybe_save_single_summary(self,
                                 iteration,
                                 ep_return,
                                 length,
                                 game_name: str,
                                 save_if_eval=False):
    prefix = 'Train/' if not self._agent.eval_mode else 'Eval/'
    if not self._agent.eval_mode or save_if_eval:
      with self._summary_writer.as_default():
        normalized_score = normalize_score(ep_return, game_name)
        tf.summary.scalar(prefix + 'EpisodeLength', length, step=iteration)
        tf.summary.scalar(prefix + 'EpisodeReturn', ep_return, step=iteration)
        tf.summary.scalar(prefix + 'EpisodeNormalizedScore', normalized_score, step=iteration)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    logging.info('Beginning training...')

    game_name_list = [
      "Breakout",
      "Boxing",
    ]

    for iteration, game_name_train in enumerate(game_name_list):
      logging.info(f'Starting iteration {iteration} {game_name_train}')
      statistics = iteration_statistics.IterationStatistics()

      # train
      self._run_train_phase(game_name_train, iteration, statistics)

      # eval
      for game_name_eval in game_name_list:
        self._run_eval_phase(game_name_eval, iteration, statistics)

      statistics = statistics.data_lists
      self._log_experiment(iteration, statistics=statistics)
      self._summary_writer.flush()


def delete_ind_from_array(array, ind, axis=0):
  start = tuple(([slice(None)] * axis) + [slice(0, ind)])
  end = tuple(([slice(None)] * axis) + [slice(ind + 1, array.shape[axis] + 1)])
  tensor = np.concatenate([array[start], array[end]], axis)
  return tensor
