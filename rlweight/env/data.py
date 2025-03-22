import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass

TICKER = str
FACTOR = pd.DataFrame
CHANGE = pd.DataFrame
TARVOL = pd.DataFrame


@dataclass
class Data:
    columns: list[str]
    index: pd.DatetimeIndex
    factor: np.ndarray
    change: np.ndarray
    tarvol: np.ndarray

    def __len__(self):
        return len(self.index)


class DataTransformer:
    def __init__(self):
        pass

    def transform(
        self, close_dict: Dict[TICKER, pd.Series], window: int, target_vol: float
    ) -> Tuple[FACTOR, CHANGE, TARVOL]:
        """
        Transform Data from close_dict

        close_dict: {ticker:close_series}
        """

        values = {}
        closes = {}

        # Calculate Sharpe Momentum Weight
        for ticker, close in close_dict.items():
            sharpe_weight = self._calculate_sharpe_momentum_weight(close, window=window)

            values[ticker] = sharpe_weight
            closes[ticker] = close

        values = pd.DataFrame(values).dropna()
        closes = pd.DataFrame(closes).dropna()
        change = closes.pct_change(fill_method=None).shift(-1).dropna()
        tarvol = target_vol / change.rolling(window).std().dropna()

        # Neutralize
        values = values.apply(lambda x: x - np.mean(x), axis=1)
        values = values.apply(lambda x: x / np.abs(x).sum(), axis=1)

        # Indexing
        common_index = values.index.intersection(change.index)
        common_index = common_index.intersection(tarvol.index)

        tarvol = tarvol.loc[common_index]
        change = change.loc[common_index]
        values = values.loc[common_index]

        return values, change, tarvol

    def _cdf(self, value):
        """
        Cumulative Distribution Function
        """
        return 0.5 * (1 + math.erf(value / math.sqrt(2)))

    def _calculate_sharpe_momentum_weight(self, close: pd.Series, window: int):
        """
        Calculate the Sharpe Momentum Weight
        """
        mean = close.pct_change().rolling(window).mean()
        vol = close.pct_change().rolling(window).std()
        sharpe = mean / vol
        momentum = sharpe.apply(lambda x: 2 * self._cdf(x) - 1)
        return momentum
