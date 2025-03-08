from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_tickers: int
    dropout_rate: float
