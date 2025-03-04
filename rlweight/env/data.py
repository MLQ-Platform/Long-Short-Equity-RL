import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Data:
    columns: list[str]
    index: pd.DatetimeIndex
    factor: np.ndarray
    change: np.ndarray

    def __len__(self):
        return len(self.index)
