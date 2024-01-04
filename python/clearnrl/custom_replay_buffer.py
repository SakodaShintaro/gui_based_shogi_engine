import warnings
from typing import Any, Dict, List, Optional, Union, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.buffers import BaseBuffer


class CustomBufferSamples(NamedTuple):
    """
    通常のReplayBufferから変更
    obs (bs, C, H, W) -> (bs, T, C, H, W)
    act (bs, 1) -> (bs, T, 1)
    nxt (bs, C, H, W) -> 削除(obsのTターン目)
    don (bs, 1) -> (bs, T, 1)
    rew (bs, 1) -> (bs, T, 1)
    """
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    total_times: th.Tensor


class CustomBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    total_times: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str],
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=1)

        self.seq_len = 20

        # Adjust buffer size
        self.buffer_size = max(buffer_size, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.observations = np.zeros(
            (self.buffer_size, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.action_dim), dtype=self._maybe_cast_dtype(
                action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.float32)
        self.total_times = np.zeros((self.buffer_size), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes +
                self.rewards.nbytes + self.dones.nbytes
            )

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )
        self.total_time = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((1, *self.obs_shape))
            next_obs = next_obs.reshape((1, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((1, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        self.observations[(self.pos + 1) %
                          self.buffer_size] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.total_times[self.pos] = np.array(self.total_time)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
        self.total_time += 1

    def sample(self, batch_size: int) -> CustomBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size - self.seq_len,
                          size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(
                0, self.pos - self.seq_len, size=batch_size)
        return self._get_samples(batch_inds)

    def get_latest(self) -> CustomBufferSamples:
        batch_inds = np.array([self.pos - self.seq_len])
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> CustomBufferSamples:
        data_list = []
        for _ in range(self.seq_len):
            data = (
                self._normalize_obs(self.observations[batch_inds, :]),
                self.actions[batch_inds, :],
                self._normalize_obs(
                    self.observations[(batch_inds + 1) % self.buffer_size, :]),
                self.dones[batch_inds].reshape(-1, 1),
                self._normalize_reward(
                    self.rewards[batch_inds].reshape(-1, 1)),
                self.total_times[batch_inds].reshape(-1, 1),
            )
            data_list.append(CustomBufferSamples(
                *tuple(map(self.to_torch, data))))
            batch_inds = (batch_inds + 1) % self.buffer_size
        data = CustomBufferSamples(
            *tuple(map(lambda x: th.stack(x, dim=1), zip(*data_list))))
        return data

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype
