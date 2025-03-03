import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.storage = deque(maxlen=max_size)

    def add(self, transition):
        """
        배치 데이터를 개별 트랜지션 단위로 버퍼에 저장
        """
        self.storage.append(transition)

    def __len__(self):
        return len(self.storage)

    def sample(self, batch_size):
        """
        버퍼에서 batch_size만큼 샘플링하여 반환
        """
        batch = random.sample(self.storage, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.vstack(states),
            np.vstack(actions),
            np.array(rewards).reshape(-1, 1),
            np.vstack(next_states),
            np.array(dones).reshape(-1, 1),
        )
