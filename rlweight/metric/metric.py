import numpy as np
import pandas as pd


class Metric:
    """
    Strategy Performance Metric Calculator
    """

    def __init__(self, portfolio_value: pd.Series):
        self.portfolio_value = portfolio_value

    def evaluate(self):
        """
        Final Result
        """
        return {
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "mdd": self._calculate_maxiumum_drawdown(),
            "win_rate": self._calculate_win_rate(),
            "pnl_rate": self._calculate_pnl_rate(),
            "total_loss": self._calculate_total_loss(),
            "total_profit": self._calculate_total_profit(),
            "final_margin": self._calculate_final_margin(),
            "cumulative_return": self._calculate_cumulative_return(),
        }

    def _calculate_cumulative_return(self):
        """
        calculating cumulative return
        """
        pv = self.portfolio_value
        return round(pv.iloc[-1] / pv.iloc[0] - 1, 4)

    def _calculate_final_margin(self):
        """
        calculating final margin
        """
        pv = self.portfolio_value
        return round(pv.iloc[-1], 4)

    def _calculate_sharpe_ratio(self):
        """
        calculating sharpe ratio
        """
        pv = self.portfolio_value

        ret = (pv.iloc[1:] / pv.iloc[:-1]) - 1
        sr = np.mean(ret) / (np.std(ret) + 1e-10)
        return round(sr, 4)

    def _calculate_maxiumum_drawdown(self):
        """
        calculating mdd
        """
        pv = self.portfolio_value

        rolling_max = np.maximum.accumulate(pv)
        # Drawdown
        drawdowns = (rolling_max - pv) / rolling_max
        # Maximum Drawdown
        mdd = np.max(drawdowns)
        return round(mdd, 4)

    def _calculate_win_rate(self):
        """
        calculating win rate
        """
        pv = self.portfolio_value

        diff = pv.diff()

        rate = diff[diff > 0].shape[0] / diff.shape[0]
        return round(rate, 4)

    def _calculate_total_profit(self):
        """
        calculating total profit
        """
        pv = self.portfolio_value

        diff = pv.diff()

        total_profit = sum(diff[diff > 0])
        return round(total_profit, 4)

    def _calculate_total_loss(self):
        """
        calculating total loss
        """
        pv = self.portfolio_value

        diff = pv.diff()

        total_loss = sum(diff[diff < 0])
        return round(total_loss, 4)

    def _calculate_pnl_rate(self):
        """
        calculating pnl rate
        """
        if self._calculate_total_loss() == 0:
            return np.inf

        rate = self._calculate_total_profit() / (self._calculate_total_loss() + 1e-10)
        return round(abs(rate), 4)
