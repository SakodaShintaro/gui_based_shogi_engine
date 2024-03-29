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

"""Implementation of BiggerBetterFaster (BBF) and SR-SPR in JAX."""

import collections
import copy
import functools
import itertools
import time

from absl import logging
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.rainbow import rainbow_agent as dopamine_rainbow_agent
from dopamine.replay_memory import prioritized_replay_buffer
from flax.core.frozen_dict import FrozenDict
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf

from bbf import spr_networks
from bbf.replay_memory import subsequence_replay_buffer


def prefetch_to_device(iterator, size):
  """Shard and prefetch batches on device.

  This utility takes an iterator and returns a new iterator which fills an on
  device prefetch buffer. Eager prefetching can improve the performance of
  training loops significantly by overlapping compute and data transfer.

  This utility is mostly useful for GPUs, for TPUs and CPUs it should not be
  necessary -- the TPU & CPU memory allocators (normally) don't pick a memory
  location that isn't free yet so they don't block. Instead those allocators
  OOM.

  Args:
    iterator: an iterator that yields a pytree of ndarrays where the first
      dimension is sharded across devices.
    size: the size of the prefetch buffer.  If you're training on GPUs, 2 is
      generally the best choice because this guarantees that you can overlap a
      training step on GPU with a data prefetch step on CPU.

  Yields:
    The original items from the iterator where each ndarray is now a sharded to
    the specified devices.
  """
  queue = collections.deque()

  def enqueue(n):
    for data in itertools.islice(iterator, n):
      queue.append(jax.device_put(data, device=jax.local_devices()[0]))

  enqueue(size)  # Fill up the buffer.
  while queue:
    yield queue.popleft()
    enqueue(1)


def copy_params(source, target, keys=("encoder", "transition_model")):
  """Copies a set of keys from one set of params to another.

  Args:
    source: Set of parameters to take keys from.
    target: Set of parameters to overwrite keys in.
    keys: Set of keys to copy.

  Returns:
    A parameter dictionary of the same shape as target.
  """
  if (
      isinstance(source, dict)
      or isinstance(source, collections.OrderedDict)
      or isinstance(source, FrozenDict)
  ):
    fresh_dict = {}
    for k, v in source.items():
      if k in keys:
        fresh_dict[k] = v
      else:
        fresh_dict[k] = copy_params(source[k], target[k], keys)
    return fresh_dict
  else:
    return target


@functools.partial(jax.jit, static_argnames=("keys", "strip_params_layer"))
def interpolate_weights(
    old_params,
    new_params,
    keys,
    old_weight=0.5,
    new_weight=0.5,
    strip_params_layer=True,
):
  """Interpolates between two parameter dictionaries.

  Args:
    old_params: The first parameter dictionary.
    new_params: The second parameter dictionary, of same shape and structure.
    keys: Which keys in the parameter dictionaries to interpolate. If None,
      interpolates everything.
    old_weight: The weight to place on the old dictionary.
    new_weight: The weight to place on the new dictionary.
    strip_params_layer: Whether to strip an outer "params" layer, as is often
      present in e.g., Flax.

  Returns:
    A parameter dictionary of the same shape as the inputs.
  """
  if strip_params_layer:
    old_params = old_params["params"]
    new_params = new_params["params"]

  def combination(old_param, new_param):
    return old_param * old_weight + new_param * new_weight

  combined_params = {}
  if keys is None:
    keys = old_params.keys()

  old_params = FrozenDict(old_params)
  new_params = FrozenDict(new_params)

  for k in keys:
    combined_params[k] = jax.tree_util.tree_map(combination, old_params[k],
                                                new_params[k])
  for k, v in old_params.items():
    if k not in keys:
      combined_params[k] = v

  if strip_params_layer:
    combined_params = {"params": combined_params}
  return FrozenDict(combined_params)


@functools.partial(
    jax.jit,
    static_argnames=(
        "state_shape",
        "keys_to_copy",
        "shrink_perturb_keys",
        "network_def",
        "optimizer",
    ),
)
def jit_reset(
    online_params,
    target_network_params,
    optimizer_state,
    network_def,
    optimizer,
    rng,
    state_shape,
    support,
    shrink_perturb_keys,
    shrink_factor,
    perturb_factor,
    keys_to_copy,
):
  """A jittable function to reset network parameters.

  Args:
    online_params: Parameter dictionary for the online network.
    target_network_params: Parameter dictionary for the target network.
    optimizer_state: Optax optimizer state.
    network_def: Network definition.
    optimizer: Optax optimizer.
    rng: JAX PRNG key.
    state_shape: Shape of the network inputs.
    do_rollout: Whether to do a dynamics model rollout (e.g., if SPR is being
      used).
    support: Support of the categorical distribution if using distributional RL.
    shrink_perturb_keys: Parameter keys to apply shrink-and-perturb to.
    shrink_factor: Factor to rescale current weights by (1 keeps , 0 deletes).
    perturb_factor: Factor to scale random noise by in [0, 1].
    keys_to_copy: Keys to copy over without resetting.

  Returns:
  """
  online_rng, target_rng = jax.random.split(rng, 2)
  state = jnp.zeros(state_shape, dtype=jnp.float32)
  # Create some dummy actions of arbitrary length to initialize the transition
  # model, if the network has one.
  actions = jnp.zeros((5,))
  random_params = network_def.init(
      x=state,
      actions=actions,
      do_rollout=True,
      rngs={
          "params": online_rng,
          "dropout": rng
      },
      support=support,
  )
  target_random_params = network_def.init(
      x=state,
      actions=actions,
      do_rollout=True,
      rngs={
          "params": target_rng,
          "dropout": rng
      },
      support=support,
  )

  if shrink_perturb_keys:
    online_params = interpolate_weights(
        online_params,
        random_params,
        shrink_perturb_keys,
        old_weight=shrink_factor,
        new_weight=perturb_factor,
    )
  online_params = FrozenDict(
      copy_params(online_params, random_params, keys=keys_to_copy))

  updated_optim_state = []
  optim_state = optimizer.init(online_params)
  for i in range(len(optim_state)):
    optim_to_copy = copy_params(
        dict(optimizer_state[i]._asdict()),
        dict(optim_state[i]._asdict()),
        keys=keys_to_copy,
    )
    optim_to_copy = FrozenDict(optim_to_copy)
    updated_optim_state.append(optim_state[i]._replace(**optim_to_copy))
  optimizer_state = tuple(updated_optim_state)

  if shrink_perturb_keys:
    target_network_params = interpolate_weights(
        target_network_params,
        target_random_params,
        shrink_perturb_keys,
        old_weight=shrink_factor,
        new_weight=perturb_factor,
    )
  target_network_params = copy_params(
      target_network_params, target_random_params, keys=keys_to_copy)
  target_network_params = FrozenDict(target_network_params)

  return online_params, target_network_params, optimizer_state, random_params


def exponential_decay_scheduler(
    decay_period, warmup_steps, initial_value, final_value, reverse=False
):
  """Instantiate a logarithmic schedule for a parameter.

  By default the extreme point to or from which values decay logarithmically
  is 0, while changes near 1 are fast. In cases where this may not
  be correct (e.g., lambda) pass reversed=True to get proper
  exponential scaling.

  Args:
      decay_period: float, the period over which the value is decayed.
      warmup_steps: int, the number of steps taken before decay starts.
      initial_value: float, the starting value for the parameter.
      final_value: float, the final value for the parameter.
      reverse: bool, whether to treat 1 as the asmpytote instead of 0.

  Returns:
      A decay function mapping step to parameter value.
  """
  if reverse:
    initial_value = 1 - initial_value
    final_value = 1 - final_value

  start = onp.log(initial_value)
  end = onp.log(final_value)

  if decay_period == 0:
    return lambda x: initial_value if x < warmup_steps else final_value

  def scheduler(step):
    steps_left = decay_period + warmup_steps - step
    bonus_frac = steps_left / decay_period
    bonus = onp.clip(bonus_frac, 0.0, 1.0)
    new_value = bonus * (start - end) + end

    new_value = onp.exp(new_value)
    if reverse:
      new_value = 1 - new_value
    return new_value

  return scheduler


def get_lambda_weights(l, horizon):
  weights = jnp.ones((horizon - 1,)) * l
  weights = jnp.cumprod(weights) * (1 - l) / (l)
  weights = jnp.concatenate([weights, jnp.ones((1,)) * (1 - jnp.sum(weights))])
  return weights


@jax.jit
def tree_norm(tree):
  return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))


@functools.partial(jax.jit)
def jit_split(rng):
  return jax.random.split(rng, 2)


@functools.partial(
    jax.jit,
    static_argnums=(0, 4, 5, 6, 7, 8, 10, 11, 13)
)
def select_action(
    network_def,
    params,
    state,
    rng,
    num_actions,
    eval_mode,
    epsilon_eval,
    epsilon_train,
    epsilon_decay_period,
    training_steps,
    min_replay_history,
    epsilon_fn,
    support,
    no_noise,
):
  """Select an action from the set of available actions."""

  rng, rng1 = jax.random.split(rng)
  state = spr_networks.process_inputs(
      state, rng=rng1, data_augmentation=False, dtype=jnp.float32
  )

  epsilon = jnp.where(
      eval_mode,
      epsilon_eval,
      epsilon_fn(
          epsilon_decay_period,
          training_steps,
          min_replay_history,
          epsilon_train,
      ),
  )

  def q_online(state, key, actions=None, do_rollout=False):
    return network_def.apply(
        params,
        state,
        actions=actions,
        do_rollout=do_rollout,
        key=key,
        support=support,
        rngs={"dropout": key},
        mutable=["batch_stats"],
        eval_mode=no_noise,
    )

  rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)
  p = jax.random.uniform(rng1, shape=(state.shape[0],))
  rng2 = jax.random.split(rng2, state.shape[0])
  q_values = get_q_values_no_actions(q_online, state, rng2)

  best_actions = jnp.argmax(q_values, axis=-1)
  new_actions = jnp.where(
      p <= epsilon,
      jax.random.randint(
          rng3,
          (state.shape[0],),
          0,
          num_actions,
      ),
      best_actions,
  )
  return rng, new_actions


@functools.partial(jax.vmap, in_axes=(None, 0, 0), axis_name="batch")
def get_q_values_no_actions(model, states, rng):
  results = model(states, actions=None, do_rollout=False, key=rng)[0]
  return results.q_values


@functools.partial(jax.vmap, in_axes=(None, 0, 0, None, 0), axis_name="batch")
def get_logits(model, states, actions, do_rollout, rng):
  results = model(states, actions=actions, do_rollout=do_rollout, key=rng)[0]
  return results.logits, results.latent, results.representation


@functools.partial(jax.vmap, in_axes=(None, 0, 0, None, 0), axis_name="batch")
def get_q_values(model, states, actions, do_rollout, rng):
  results = model(states, actions=actions, do_rollout=do_rollout, key=rng)[0]
  return results.q_values, results.latent, results.representation


@functools.partial(jax.vmap, in_axes=(None, 0, 0), axis_name="batch")
@functools.partial(jax.vmap, in_axes=(None, 0, None), axis_name="time")
def get_spr_targets(model, states, key):
  results = model(states, key)
  return results


train_static_argnums = (
    0,
    3,
    15,
)


def train(
    network_def,  # 0, static
    online_params,  # 1
    target_params,  # 2
    optimizer,  # 3, static
    optimizer_state,  # 4
    raw_states,  # 5
    actions,  # 6
    raw_next_states,  # 7
    rewards,  # 8
    terminals,  # 9
    same_traj_mask,  # 10
    loss_weights,  # 11
    support,  # 12
    cumulative_gamma,  # 13
    rng,  # 14
    batch_size,  # 15, static
    target_update_tau,  # 16
    target_update_every,  # 17
    step,  # 18
):
  """Run one or more training steps for BBF.

  Args:
    network_def: Network definition class.
    online_params: Online parameter dictionary.
    target_params: Target parameter dictionary.
    optimizer: Optax optimizer.
    optimizer_state: Optax optimizer state.
    raw_states: Raw state inputs (not preprocessed, uint8), (B, T, H, W, C).
    actions: Actions (int32), (B, T).
    raw_next_states: Raw inputs for states at s_{t+n}, (B, T, H, W, C).
    rewards: Rewards, (B, T).
    terminals: Terminal signals, (B, T).
    same_traj_mask: Mask denoting valid continuations of trajectories, (B, T).
    loss_weights: Loss weights from prioritized replay sampling, (B,).
    support: support for the categorical distribution in C51
    cumulative_gamma: Discount factors (B,), gamma^n for the current n, gamma.
    rng: JAX PRNG Key.
    dtype: Jax dtype for training (float32, float16, or bfloat16)
    batch_size: int, size of each batch to run. Must cleanly divide the leading
      axis of input arrays. If smaller, the function will chain together
      multiple batches.
    target_update_tau: Float in [0, 1], tau for target network updates. 1 is
      hard (online target), 0 is frozen.
    target_update_every: How often to do a target update (in gradient steps).
    step: The current gradient step.

  Returns:
    Updated online params, target params, optimizer state, dynamic scale,
    and dictionary of metrics.
  """

  @functools.partial(jax.jit,
                     donate_argnums=(0,),
                     )
  def train_one_batch(state, inputs):
    """Runs a training step."""
    # Unpack inputs from scan
    (
        online_params,
        target_params,
        optimizer_state,
        rng,
        step,
    ) = state
    (
        raw_states,
        actions,
        raw_next_states,
        rewards,
        terminals,
        same_traj_mask,
        loss_weights,
        cumulative_gamma,
    ) = inputs
    same_traj_mask = same_traj_mask[:, 1:]
    rewards = rewards[:, 0]
    terminals = terminals[:, 0]
    cumulative_gamma = cumulative_gamma[:, 0]

    rng, rng1, rng2 = jax.random.split(rng, num=3)
    states = spr_networks.process_inputs(
        raw_states, rng=rng1, data_augmentation=True, dtype=jnp.float32
    )
    next_states = spr_networks.process_inputs(
        raw_next_states[:, 0],
        rng=rng2,
        data_augmentation=True,
        dtype=jnp.float32,
    )
    current_state = states[:, 0]

    # Split the current rng to update the rng after this call
    rng, rng1, rng2 = jax.random.split(rng, num=3)

    batch_rngs = jax.random.split(rng, num=states.shape[0])
    target_rng = batch_rngs

    def q_online(state, key, actions=None, do_rollout=False):
      return network_def.apply(
          online_params,
          state,
          actions=actions,
          do_rollout=do_rollout,
          key=key,
          rngs={"dropout": key},
          support=support,
          mutable=["batch_stats"],
      )

    def q_target(state, key):
      return network_def.apply(
          target_params,
          state,
          key=key,
          support=support,
          eval_mode=False,
          rngs={"dropout": key},
          mutable=["batch_stats"],
      )

    def encode_project(state, key):
      return network_def.apply(
          target_params,
          state,
          key=key,
          rngs={"dropout": key},
          eval_mode=True,
          method=network_def.encode_project,
      )

    def loss_fn(
        params,
        target,
        spr_targets,
        loss_multipliers,
    ):
      """Computes the distributional loss for C51."""

      def q_online(state, key, actions=None, do_rollout=False):
        return network_def.apply(
            params,
            state,
            actions=actions,
            do_rollout=do_rollout,
            key=key,
            rngs={"dropout": key},
            support=support,
            mutable=["batch_stats"],
        )

      (logits, spr_predictions, _) = get_logits(
          q_online, current_state, actions[:, :-1], True, batch_rngs
      )
      logits = jnp.squeeze(logits)
      # Fetch the logits for its selected action. We use vmap to perform this
      # indexing across the batch.
      chosen_action_logits = jax.vmap(lambda x, y: x[y])(
          logits, actions[:, 0]
      )
      dqn_loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
          target, chosen_action_logits)
      td_error = dqn_loss + jnp.nan_to_num(target * jnp.log(target)).sum(-1)

      spr_predictions = spr_predictions.transpose(1, 0, 2)
      spr_predictions = spr_predictions / jnp.linalg.norm(
          spr_predictions, 2, -1, keepdims=True)
      spr_targets = spr_targets / jnp.linalg.norm(
          spr_targets, 2, -1, keepdims=True)
      spr_loss = jnp.power(spr_predictions - spr_targets, 2).sum(-1)
      spr_loss = (spr_loss * same_traj_mask.transpose(1, 0)).mean(0)

      spr_weight = 5
      loss = dqn_loss + spr_weight * spr_loss
      loss = loss_multipliers * loss

      mean_loss = jnp.mean(loss)

      aux_losses = {
          "TotalLoss": jnp.mean(mean_loss),
          "DQNLoss": jnp.mean(dqn_loss),
          "TD Error": jnp.mean(td_error),
          "SPRLoss": jnp.mean(spr_loss),
      }

      return mean_loss, (aux_losses)

    # Use the weighted mean loss for gradient computation.
    target = target_output(
        q_online,
        q_target,
        next_states,
        rewards,
        terminals,
        support,
        cumulative_gamma,
        target_rng,
    )
    target = jax.lax.stop_gradient(target)

    future_states = states[:, 1:]
    spr_targets = get_spr_targets(encode_project, future_states, target_rng)
    spr_targets = spr_targets.transpose(1, 0, 2)

    # Get the unweighted loss without taking its mean for updating priorities.

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, aux_losses), grad = grad_fn(
        online_params,
        target,
        spr_targets,
        loss_weights,
    )

    grad_norm = tree_norm(grad)
    aux_losses["GradNorm"] = grad_norm
    updates, new_optimizer_state = optimizer.update(
        grad, optimizer_state, params=online_params)
    new_online_params = optax.apply_updates(online_params, updates)

    optimizer_state = new_optimizer_state
    online_params = new_online_params

    target_update_step = functools.partial(
        interpolate_weights,
        keys=None,
        old_weight=1 - target_update_tau,
        new_weight=target_update_tau,
    )
    target_params = jax.lax.cond(
        step % target_update_every == 0,
        target_update_step,
        lambda old, new: old,
        target_params,
        online_params,
    )

    return (
        (
            online_params,
            target_params,
            optimizer_state,
            rng2,
            step + 1,
        ),
        aux_losses,
    )

  init_state = (
      online_params,
      target_params,
      optimizer_state,
      rng,
      step,
  )
  assert raw_states.shape[0] % batch_size == 0
  num_batches = raw_states.shape[0] // batch_size

  inputs = (
      raw_states.reshape(num_batches, batch_size, *raw_states.shape[1:]),
      actions.reshape(num_batches, batch_size, *actions.shape[1:]),
      raw_next_states.reshape(
          num_batches, batch_size, *raw_next_states.shape[1:]
      ),
      rewards.reshape(num_batches, batch_size, *rewards.shape[1:]),
      terminals.reshape(num_batches, batch_size, *terminals.shape[1:]),
      same_traj_mask.reshape(
          num_batches, batch_size, *same_traj_mask.shape[1:]
      ),
      loss_weights.reshape(num_batches, batch_size, *loss_weights.shape[1:]),
      cumulative_gamma.reshape(
          num_batches, batch_size, *cumulative_gamma.shape[1:]
      ),
  )

  (
      (
          online_params,
          target_params,
          optimizer_state,
          rng,
          step,
      ),
      aux_losses,
  ) = jax.lax.scan(train_one_batch, init_state, inputs)

  return (
      online_params,
      target_params,
      optimizer_state,
      {k: jnp.reshape(v, (-1,)) for k, v in aux_losses.items()},
  )


@functools.partial(
    jax.vmap,
    in_axes=(None, None, 0, 0, 0, None, 0, 0),
    axis_name="batch",
)
def target_output(
    model,
    target_network,
    next_states,
    rewards,
    terminals,
    support,
    cumulative_gamma,
    rng,
):
  """Builds the C51 target distribution or DQN target Q-values."""
  is_terminal_multiplier = 1.0 - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier

  target_dist, _ = target_network(next_states, key=rng)
  online_dist, _ = model(next_states, key=rng)

  # Action selection using Q-values for next-state
  q_values = jnp.squeeze(online_dist.q_values)
  next_qt_argmax = jnp.argmax(q_values)

  # Compute the target Q-value distribution
  probabilities = jnp.squeeze(target_dist.probabilities)
  next_probabilities = probabilities[next_qt_argmax]
  target_support = rewards + gamma_with_terminal * support
  target = dopamine_rainbow_agent.project_distribution(
      target_support, next_probabilities, support)

  return jax.lax.stop_gradient(target)


@gin.configurable
def create_scaling_optimizer(
    learning_rate=6.25e-5,
    beta1=0.9,
    beta2=0.999,
    eps=1.5e-4,
    weight_decay=0.0,
):
  logging.info(
      f"Creating AdamW optimizer with settings lr={learning_rate}, beta1={beta1}, "
      f"beta2={beta2}, eps={eps}, wd={weight_decay}")
  mask = lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p)
  return optax.adamw(
      learning_rate,
      b1=beta1,
      b2=beta2,
      eps=eps,
      weight_decay=weight_decay,
      mask=mask,
  )


@gin.configurable
class BBFAgent(dqn_agent.JaxDQNAgent):
  """A compact implementation of the full Rainbow agent."""

  def __init__(
      self,
      num_actions,
      num_updates_per_train_step=1,
      jumps=0,
      batch_size=32,
      replay_ratio=64,
      batches_to_group=1,
      update_horizon=10,
      max_update_horizon=None,
      min_gamma=None,
      reset_every=-1,
      no_resets_after=-1,
      learning_rate=0.0001,
      encoder_learning_rate=0.0001,
      shrink_perturb_keys="",
      perturb_factor=0.2,  # original was 0.1
      shrink_factor=0.8,  # original was 0.4
      target_update_tau=1.0,
      cycle_steps=0,
      target_update_period=1,
      offline_update_frac=0,
      summary_writer=None,
      seed=None,
  ):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      num_updates_per_train_step: int, Number of gradient updates every training
        step. Defaults to 1.
      jumps: int, number of steps to predict in SPR.
      batch_size: number of examples per batch.
      replay_ratio: Average number of times an example is replayed during
        training. Divide by batch_size to get the 'replay ratio' definition
        based on gradient steps by D'Oro et al.
      batches_to_group: Number of batches to group together into a single jit.
      update_horizon: int, n-step return length.
      max_update_horizon: int, n-step start point for annealing.
      min_gamma: float, gamma start point for annealing.
      reset_every: int, how many training steps between resets. 0 to disable.
      no_resets_after: int, training step to cease resets before.
      learning_rate: Learning rate for all non-encoder parameters.
      encoder_learning_rate: Learning rate for the encoder (if different).
      shrink_perturb_keys: string of comma-separated keys, such as
        'encoder,transition_model', to which to apply shrink & perturb to.
      perturb_factor: float, weight of random noise in shrink & perturb.
      shrink_factor: float, weight of initial parameters in shrink & perturb.
      target_update_tau: float, update parameter for EMA target network.
      cycle_steps: int, number of steps to anneal hyperparameters after reset.
      target_update_period: int, steps per target network update.
      offline_update_frac: float, fraction of a reset interval to do offline
        after each reset to warm-start the new network. summary_writer=None,
      summary_writer: SummaryWriter object, for outputting training statistics.
      seed: int, a seed for Jax RNG and initialization.
    """
    logging.info(
        "Creating %s agent with the following parameters:",
        self.__class__.__name__,
    )
    logging.info("\t num_updates_per_train_step: %d",
                 num_updates_per_train_step)
    # We need casting because passing arguments can convert ints to floats
    num_atoms=51
    vmax = 10.0
    vmin = -vmax
    self._support = jnp.linspace(vmin, vmax, num_atoms)
    self._replay_ratio = int(replay_ratio)
    self._batch_size = int(batch_size)
    self._batches_to_group = int(batches_to_group)
    self.update_horizon = int(update_horizon)
    self._jumps = int(jumps)
    self.log_every = 100

    self.reset_every = int(reset_every)
    self.offline_update_frac = float(offline_update_frac)
    self.no_resets_after = int(no_resets_after)
    self.cumulative_resets = 0
    self.next_reset = self.reset_every + 1

    self.learning_rate = learning_rate
    self.encoder_learning_rate = encoder_learning_rate

    self.shrink_perturb_keys = [
        s for s in shrink_perturb_keys.lower().split(",") if s
    ]
    self.shrink_perturb_keys = tuple(self.shrink_perturb_keys)
    self.shrink_factor = shrink_factor
    self.perturb_factor = perturb_factor

    self.grad_steps = 0
    self.cycle_grad_steps = 0
    self.target_update_period = int(target_update_period)
    self.target_update_tau = target_update_tau

    if max_update_horizon is None:
      self.max_update_horizon = self.update_horizon
      self.update_horizon_scheduler = lambda x: self.update_horizon
    else:
      self.max_update_horizon = int(max_update_horizon)
      n_schedule = exponential_decay_scheduler(
          cycle_steps, 0, 1, self.update_horizon / self.max_update_horizon
      )
      self.update_horizon_scheduler = lambda x: int(  # pylint: disable=g-long-lambda
          onp.round(n_schedule(x) * self.max_update_horizon)
      )

    logging.info("\t Found following local devices: %s",
                 str(jax.local_devices()))

    self.dtype = jnp.float32
    super().__init__(
        num_actions=num_actions,
        network=functools.partial(
            spr_networks.RainbowDQNNetwork,
            num_atoms=num_atoms,
            dtype=self.dtype,
        ),
        epsilon_fn=dqn_agent.linearly_decaying_epsilon,
        target_update_period=self.target_update_period,
        update_horizon=self.max_update_horizon,
        summary_writer=summary_writer,
        seed=seed,
    )

    self.set_replay_settings()

    if min_gamma is None or cycle_steps <= 1:
      self.min_gamma = self.gamma
      self.gamma_scheduler = lambda x: self.gamma
    else:
      self.min_gamma = min_gamma
      self.gamma_scheduler = exponential_decay_scheduler(
          cycle_steps, 0, self.min_gamma, self.gamma, reverse=True
      )

    self.cumulative_gamma = (onp.ones(
        (self.max_update_horizon,)) * self.gamma).cumprod()

    self.train_fn = jax.jit(train, static_argnums=train_static_argnums,
                            device=jax.local_devices()[0])

  def _build_networks_and_optimizer(self):
    self._rng, rng = jax.random.split(self._rng)
    self.state_shape = self.state.shape

    # Create some dummy actions of arbitrary length to initialize the transition
    # model, if the network has one.
    actions = jnp.zeros((5,))
    self.online_params = self.network_def.init(
        x=self.state.astype(self.dtype),
        actions=actions,
        do_rollout=True,
        rngs={
            "params": rng,
            "dropout": rng
        },
        support=self._support,
    )
    optimizer = create_scaling_optimizer(
        learning_rate=self.learning_rate,
    )
    encoder_optimizer = create_scaling_optimizer(
        learning_rate=self.encoder_learning_rate,
    )

    encoder_keys = {"encoder", "transition_model"}
    self.encoder_mask = FrozenDict({
        "params": {k: k in encoder_keys for k in self.online_params["params"]}
    })
    self.head_mask = FrozenDict({
        "params": {
            k: k not in encoder_keys for k in self.online_params["params"]
        }
    })

    self.optimizer = optax.chain(
        optax.masked(encoder_optimizer, self.encoder_mask),
        optax.masked(optimizer, self.head_mask),
    )

    self.online_params = FrozenDict(self.online_params)
    self.optimizer_state = self.optimizer.init(self.online_params)
    self.target_network_params = copy.deepcopy(self.online_params)
    self.random_params = copy.deepcopy(self.online_params)

    self.online_params = jax.device_put(
        self.online_params, jax.local_devices()[0]
    )
    self.target_params = jax.device_put(
        self.target_network_params, jax.local_devices()[0]
    )
    self.random_params = jax.device_put(
        self.random_params, jax.local_devices()[0]
    )
    self.optimizer_state = jax.device_put(
        self.optimizer_state, jax.local_devices()[0]
    )

  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""
    return subsequence_replay_buffer.PrioritizedJaxSubsequenceParallelEnvReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        update_horizon=self.max_update_horizon,
        gamma=self.gamma,
        subseq_len=self._jumps + 1,
        batch_size=self._batch_size,
        observation_dtype=self.observation_dtype,
    )

  def set_replay_settings(self):
    logging.info(f"\t batch size {self._batch_size} and replay ratio {self._replay_ratio}")
    self._num_updates_per_train_step = max(1, self._replay_ratio // self._batch_size)
    self.update_period = max(1, self._batch_size // self._replay_ratio)
    logging.info(f"\t Calculated {self._num_updates_per_train_step} updates per update phase")
    logging.info(f"\t Calculated update frequency of {self.update_period} steps")
    logging.info(f"\t Setting min_replay_history to {self.min_replay_history}")
    logging.info(f"\t Setting epsilon_decay_period to {self.epsilon_decay_period}")
    self._batches_to_group = min(self._batches_to_group,
                                 self._num_updates_per_train_step)
    assert self._num_updates_per_train_step % self._batches_to_group == 0
    self._num_updates_per_train_step = int(
        max(1, self._num_updates_per_train_step / self._batches_to_group))

    logging.info(
        "\t Running %s groups of %s batch%s per %s env step%s",
        self._num_updates_per_train_step,
        self._batches_to_group,
        "es" if self._batches_to_group > 1 else "",
        self.update_period,
        "s" if self.update_period > 1 else "",
    )

  def _replay_sampler_generator(self):
    types = self._replay.get_transition_elements()
    while True:
      self._rng, rng = jit_split(self._rng)

      samples = self._replay.sample_transition_batch(
          rng,
          batch_size=self._batch_size * self._batches_to_group,
          update_horizon=self.update_horizon_scheduler(self.cycle_grad_steps),
          gamma=self.gamma_scheduler(self.cycle_grad_steps),
      )
      replay_elements = collections.OrderedDict()
      for element, element_type in zip(samples, types):
        replay_elements[element_type.name] = element
      yield replay_elements

  def sample_eval_batch(self, batch_size, subseq_len=1):
    self._rng, rng = jit_split(self._rng)
    samples = self._replay.sample_transition_batch(
        rng, batch_size=batch_size, subseq_len=subseq_len)
    types = self._replay.get_transition_elements()
    replay_elements = collections.OrderedDict()
    for element, element_type in zip(samples, types):
      replay_elements[element_type.name] = element
    # Add code for data augmentation.

    return replay_elements

  def initialize_prefetcher(self):
    self.prefetcher = prefetch_to_device(self._replay_sampler_generator(), 2)

  def _sample_from_replay_buffer(self):
    self.replay_elements = next(self.prefetcher)

  def reset_weights(self):
    self.cumulative_resets += 1
    interval = self.reset_every

    self.next_reset = int(interval) + self.training_steps
    if self.next_reset > self.no_resets_after + 1:
      logging.info(
          "\t Not resetting at step %s, as need at least"
          " %s before %s to recover.",
          self.training_steps, interval, self.no_resets_after
      )
      return
    else:
      logging.info("\t Resetting weights at step %s.",
                   self.training_steps)

    self._rng, reset_rng = jax.random.split(self._rng, 2)

    # These are the parameter entries that will be copied over unchanged
    # from the current dictionary to the new (randomly-initialized) one
    keys_to_copy = []
    keys_to_copy.append("encoder")
    keys_to_copy.append("transition_model")
    keys_to_copy = tuple(keys_to_copy)

    (
        self.online_params,
        self.target_network_params,
        self.optimizer_state,
        self.random_params,
    ) = jit_reset(
        self.online_params,
        self.target_network_params,
        self.optimizer_state,
        self.network_def,
        self.optimizer,
        reset_rng,
        self.state_shape,
        self._support,
        self.shrink_perturb_keys,
        self.shrink_factor,
        self.perturb_factor,
        keys_to_copy,
    )
    self.online_params = jax.device_put(
        self.online_params, jax.local_devices()[0]
    )
    self.target_params = jax.device_put(
        self.target_network_params, jax.local_devices()[0]
    )
    self.random_params = jax.device_put(
        self.random_params, jax.local_devices()[0]
    )
    self.optimizer_state = jax.device_put(
        self.optimizer_state, jax.local_devices()[0]
    )

    self.cycle_grad_steps = 0

    if self._replay.add_count > self.min_replay_history:
      offline_steps = int(interval * self.offline_update_frac *
                          self._num_updates_per_train_step)

      logging.info(
          "Running %s gradient steps after reset",
          offline_steps * self._batches_to_group,
      )
      for i in range(1, offline_steps + 1):
        self._training_step_update(i, offline=True)

  def _training_step_update(self, step_index, offline=False):
    """Gradient update during every training step."""
    should_log = (
        self.training_steps % self.log_every == 0 and not offline and
        step_index == 0)

    if not hasattr(self, "replay_elements"):
      self._sample_from_replay_buffer()

    # The original prioritized experience replay uses a linear exponent
    # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
    # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
    # suggested a fixed exponent actually performs better, except on Pong.
    probs = self.replay_elements["sampling_probabilities"]
    # Weight the loss by the inverse priorities.
    loss_weights = 1.0 / onp.sqrt(probs + 1e-10)
    loss_weights /= onp.max(loss_weights)
    indices = self.replay_elements["indices"]

    if should_log:
      eval_batch = self.sample_eval_batch(256)
      eval_states = eval_batch["state"].reshape(-1,
                                                *eval_batch["state"].shape[-3:])
      eval_actions = eval_batch["action"].reshape(-1,)
      self._rng, eval_rng = jax.random.split(self._rng, 2)
      og_actions = self.select_action(
          eval_states,
          self.online_params,
          eval_mode=True,
          force_zero_eps=True,
          rng=eval_rng,
          use_noise=False,
      )
      og_target_actions = self.select_action(
          eval_states,
          self.target_network_params,
          eval_mode=True,
          force_zero_eps=True,
          rng=eval_rng,
          use_noise=False,
      )

    self._rng, train_rng = jit_split(self._rng)
    (
        new_online_params,
        new_target_params,
        new_optimizer_state,
        aux_losses,
    ) = self.train_fn(
        self.network_def,
        self.online_params,
        self.target_network_params,
        self.optimizer,
        self.optimizer_state,
        self.replay_elements["state"],
        self.replay_elements["action"],
        self.replay_elements["next_state"],
        self.replay_elements["return"],
        self.replay_elements["terminal"],
        self.replay_elements["same_trajectory"],
        loss_weights,
        self._support,
        self.replay_elements["discount"],
        train_rng,
        self._batch_size,
        self.target_update_tau,
        self.target_update_period,
        self.grad_steps,
    )
    self.grad_steps += self._batches_to_group
    self.cycle_grad_steps += self._batches_to_group

    # Sample asynchronously while we wait for training
    self._sample_from_replay_buffer()

    # Rainbow and prioritized replay are parametrized by an exponent
    # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
    # leave it as is here, using the more direct sqrt(). Taking the square
    # root "makes sense", as we are dealing with a squared loss.  Add a
    # small nonzero value to the loss to avoid 0 priority items. While
    # technically this may be okay, setting all items to 0 priority will
    # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
    indices = onp.reshape(onp.asarray(indices), (-1,))
    dqn_loss = onp.reshape(onp.asarray(aux_losses["DQNLoss"]), (-1))
    priorities = onp.sqrt(dqn_loss + 1e-10)
    self._replay.set_priority(indices, priorities)

    if should_log:
      metrics = {
          **{k: onp.mean(v) for k, v in aux_losses.items()},
          "PNorm": float(tree_norm(new_online_params)),
      }

      new_actions = self.select_action(
          eval_states,
          new_online_params,
          eval_mode=True,
          force_zero_eps=True,
          rng=eval_rng,
          use_noise=False,
      )
      new_target_actions = self.select_action(
          eval_states,
          new_target_params,
          eval_mode=True,
          force_zero_eps=True,
          rng=eval_rng,
          use_noise=False,
      )
      online_churn = onp.mean(new_actions != og_actions)
      target_churn = onp.mean(new_target_actions != og_target_actions)
      online_off_policy_frac = onp.mean(new_actions != eval_actions)
      target_off_policy_frac = onp.mean(new_target_actions != eval_actions)
      online_target_agreement = onp.mean(new_actions == new_target_actions)
      churn_metrics = {
          "Online Churn": online_churn,
          "Target Churn": target_churn,
          "Online-Target Agreement": online_target_agreement,
          "Online Off-Policy Rate": online_off_policy_frac,
          "Target Off-Policy Rate": target_off_policy_frac,
      }
      metrics.update(**churn_metrics)

      if self.summary_writer is not None:
        with self.summary_writer.as_default():
          for k, v in metrics.items():
            tf.summary.scalar(f"Agent/{k}", v, step=self.training_steps)

    self.target_network_params = new_target_params
    self.online_params = new_online_params
    self.optimizer_state = new_optimizer_state

  def _store_transition(
      self,
      last_observation,
      action,
      reward,
      is_terminal,
      *args,
      priority=None,
      episode_end=False,
  ):
    """Stores a transition when in training mode."""
    is_prioritized = isinstance(
        self._replay,
        prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer,
    ) or isinstance(
        self._replay,
        subsequence_replay_buffer.PrioritizedJaxSubsequenceParallelEnvReplayBuffer,
    )
    if is_prioritized and priority is None:
      priority = onp.ones((last_observation.shape[0]))
      priority.fill(self._replay.sum_tree.max_recorded_priority)

    if not self.eval_mode:
      self._replay.add(
          last_observation,
          action,
          reward,
          is_terminal,
          *args,
          priority=priority,
          episode_end=episode_end,
      )

  def _sync_weights(self, tau):
    if tau >= 1 or tau < 0:
      self.target_network_params = self.online_params
    else:
      self.target_network_params = interpolate_weights(
          self.target_network_params,
          self.online_params,
          keys=None,  # all keys
          old_weight=1 - tau,
          new_weight=tau,
          strip_params_layer=True,
      )

  def _train_step(self):
    """Runs a single training step.

    Runs training if both:
        (1) A minimum number of frames have been added to the replay buffer.
        (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_network_params to target_network_params if
    training steps is a multiple of target update period.
    """
    if self._replay.add_count == self.min_replay_history:
      self.initialize_prefetcher()

    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        for i in range(self._num_updates_per_train_step):
          self._training_step_update(i, offline=False)
    if self.reset_every > 0 and self.training_steps > self.next_reset:
      self.reset_weights()

    self.training_steps += 1

  def _reset_state(self, n_envs):
    """Resets the agent state by filling it with zeros."""
    self.state = onp.zeros(n_envs, *self.state_shape)

  def _record_observation(self, observation):
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
    # Set current observation. We do the reshaping to handle environments
    # without frame stacking.
    observation = observation.squeeze(-1)
    if len(observation.shape) == len(self.observation_shape):
      self._observation = onp.reshape(observation, self.observation_shape)
    else:
      self._observation = onp.reshape(
          observation, (observation.shape[0], *self.observation_shape))
    # Swap out the oldest frame with the current frame.
    self.state = onp.roll(self.state, -1, axis=-1)
    self.state[Ellipsis, -1] = self._observation

  def reset_all(self, new_obs):
    """Resets the agent state by filling it with zeros."""
    n_envs = new_obs.shape[0]
    self.state = onp.zeros((n_envs, *self.state_shape))
    self._record_observation(new_obs)

  def reset_one(self, env_id):
    self.state[env_id].fill(0)

  def delete_one(self, env_id):
    self.state = onp.concatenate([self.state[:env_id], self.state[env_id + 1:]],
                                 0)

  def cache_train_state(self):
    self.training_state = (
        copy.deepcopy(self.state),
        copy.deepcopy(self._last_observation),
        copy.deepcopy(self._observation),
    )

  def restore_train_state(self):
    (self.state, self._last_observation, self._observation) = (
        self.training_state)

  def log_transition(self, observation, action, reward, terminal, episode_end):
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(
          self._last_observation,
          action,
          reward,
          terminal,
          episode_end=episode_end,
      )

  def select_action(
      self,
      state,
      select_params,
      eval_mode=False,
      use_noise=True,
      force_zero_eps=False,
      rng=None,
  ):
    force_rng = rng is not None
    if not force_rng:
      rng = self._rng
    new_rng, action = select_action(
        self.network_def,
        select_params,
        state,
        rng,
        self.num_actions,
        eval_mode or force_zero_eps,
        self.epsilon_eval if not force_zero_eps else 0.0,
        self.epsilon_train,
        self.epsilon_decay_period,
        self.training_steps,
        self.min_replay_history,
        self.epsilon_fn,
        self._support,
        not use_noise,
    )
    if not force_rng:
      self._rng = new_rng
    return action

  def step(self):
    """Records the most recent transition, returns the agent's next action, and trains if appropriate.
    """
    if not self.eval_mode:
      self._train_step()

    action = self.select_action(
        self.state,
        self.target_network_params,
        eval_mode=self.eval_mode,
        use_noise=not self.eval_mode,
    )
    self.action = onp.asarray(action)
    return self.action
