import numpy as np
from typing import Tuple
from dataclasses import dataclass
from rlweight.env.data import Data


@dataclass
class EventEnvConfig:
    fee: float
    gamma: float


class EventEnv:
    def __init__(self, config: EventEnvConfig, data: Data):
        self.config = config
        self.data = data

        self._index = 0

    @property
    def info(self):
        return {
            "data_len": len(self.data),
        }

    @staticmethod
    def neutralize(weight: np.ndarray):
        """
        weight: (num_tickers,)
        """
        return (weight - weight.mean()) / np.sum(np.abs(weight) + 1e-5)

    def reset(self):
        self._index = 0

        # (num_tickers, )
        target_weight = self.data.factor[self._index]
        # (num_tickers, )
        target_vol = self.data.tarvol[self._index]
        # (num_tickers, )
        holding_weight = np.zeros_like(target_weight, dtype=np.float32)

        obs = {
            "target_vol": target_vol.astype(np.float32),
            "target_weight": target_weight.astype(np.float32),
            "holding_weight": holding_weight.astype(np.float32),
        }
        return obs

    def execute(self, obs: dict, action: np.ndarray) -> Tuple[dict, float]:
        """
        obs: (num_tickers, 3)
        action: (num_tickers,): weight gap
        """
        # (num_tickers,)
        holding_weight = obs["holding_weight"]
        # (num_tickers,)
        weight = np.clip(holding_weight + action, -1.0, 1.0)
        # (num_tickers,)
        pct = self.data.change[self._index]

        ret = np.sum(weight * pct)

        turnover: float = np.sum(np.abs(weight - holding_weight))
        # (1,)
        reward: float = np.array([ret - turnover * self.config.fee], dtype=np.float32)

        self._index += 1
        target_vol = self.data.tarvol[self._index]
        # (num_tickers,)
        target_weight = self.data.factor[self._index]
        # (num_tickers,)
        holding_weight = weight
        # (1, )
        done = np.array([self._index == len(self.data) - 1], dtype=np.float32)
        # (num_tickers, 3)
        next_obs = {
            "target_vol": target_vol.astype(np.float32),
            "target_weight": target_weight.astype(np.float32),
            "holding_weight": holding_weight.astype(np.float32),
        }

        info = {
            "pct": pct,
            "ret": ret,
            "weights": weight,
            "turnover": turnover,
        }
        return next_obs, reward, done, info
