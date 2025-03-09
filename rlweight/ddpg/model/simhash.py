import numpy as np


class SimHash:
    def __init__(self, state_emb, k):
        np.random.seed(42)

        self.hash = {}
        self.A = np.random.normal(0, 1, (k, state_emb))

    def add(self, state: np.ndarray) -> None:
        """
        state: (num_tickers,)
        """
        discrete_state = self.discretization(state)

        key = str(discrete_state.tolist())
        self.hash[key] = self.hash.get(key, 0) + 1

    def count(self, state: np.ndarray) -> int:
        """
        state: (num_tickers,)
        """
        discrete_state = self.discretization(state)
        key = str(discrete_state.tolist())
        return self.hash.get(key, 0)

    def discretization(self, state: np.ndarray) -> np.ndarray:
        """
        state: (num_tickers,)
        """
        return np.sign(self.A @ state)
