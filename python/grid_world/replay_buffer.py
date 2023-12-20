from collections import namedtuple, deque
import random

Experience = namedtuple(
    'Experience', ('state', 'action', 'reward', 'value'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer_ = deque([], maxlen=capacity)
        self.seq_len_ = 2  # 1回のデータとして取り出す系列長

    def push(self, state, action, reward, value):
        self.buffer_.append(Experience(state, action, reward, value))

    def sample(self, batch_size):
        if len(self.buffer_) < self.seq_len_:
            return None

        max_index = len(self.buffer_) - self.seq_len_ + 1
        batch_size = min(batch_size, max_index)
        starts = random.sample(range(max_index), batch_size)
        samples = []
        for start in starts:
            samples.append([self.buffer_[i]
                           for i in range(start, start + self.seq_len_)])
        return samples

    def __len__(self):
        return len(self.buffer_)
