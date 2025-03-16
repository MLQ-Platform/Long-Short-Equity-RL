from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_tickers: int
    action_scale: float = 0.1
    purturb_scale: float = 0.005
