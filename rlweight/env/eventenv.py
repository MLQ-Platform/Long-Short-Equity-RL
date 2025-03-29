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

    def reset(self):
        """
        return initial observation

        obs:
            {
                "target_vol": (num_tickers, ),
                "factor_weight": (num_tickers, ),
                "holding_weight": (num_tickers, ),
            }
        """

        self._index = 0

        # (num_tickers, )
        factor_weight = self.data.factor[self._index]
        # (num_tickers, )
        target_vol = self.data.tarvol[self._index]
        # (num_tickers, )
        holding_weight = np.zeros_like(factor_weight, dtype=np.float32)

        obs = {
            "target_vol": target_vol.astype(np.float32),
            "factor_weight": factor_weight.astype(np.float32),
            "holding_weight": holding_weight.astype(np.float32),
        }
        return obs

    def execute(self, obs: dict, action: np.ndarray) -> Tuple[dict, float]:
        """
        execute an action and return next observation, reward, done, info

        next_obs:
            {
                "target_vol": (num_tickers, ),
                "target_weight": (num_tickers, ),
                "holding_weight": (num_tickers, ),
            }

        reward: (1,)
        done: (1,)
        info:
            {
                "pct": (1, ),
                "ret": (1, ),
                "weights": (num_tickers, ),
                "turnover": (1, ),
            }
        """
        # (num_tickers,)
        holding_weight = obs["holding_weight"]
        # (num_tickers,)
        weight = np.clip(holding_weight + action, -1.0, 1.0)
        # (num_tickers,)
        pct = self.data.change[self._index]
        # (1, )
        ret = np.sum(weight * pct)
        # (1, )
        turnover: float = np.sum(np.abs(weight - holding_weight))
        # (1,)
        reward: float = np.array([ret - turnover * self.config.fee], dtype=np.float32)

        self._index += 1
        target_vol = self.data.tarvol[self._index]
        # (num_tickers,)
        factor_weight = self.data.factor[self._index]
        # (num_tickers,)
        holding_weight = weight
        # (1, )
        done = np.array([self._index == len(self.data) - 1], dtype=np.float32)

        next_obs = {
            # (num_tickers,)
            "target_vol": target_vol.astype(np.float32),
            "factor_weight": factor_weight.astype(np.float32),
            "holding_weight": holding_weight.astype(np.float32),
        }

        info = {
            "pct": pct,
            "ret": ret,
            "weights": weight,
            "turnover": turnover,
        }
        return next_obs, reward, done, info
