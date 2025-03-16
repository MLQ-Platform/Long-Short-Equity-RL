from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_tickers: int
    action_scale: float = 0.1
    perturb_scale: float = 0.005
