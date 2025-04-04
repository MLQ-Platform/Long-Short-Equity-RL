import torch
import torch.nn as nn
from rlweight.bcq.model.config import ModelConfig


class Qnet(nn.Module):
    """
    Qnet for Clipped Double Q-learning
    """

    def __init__(self, config: ModelConfig):
        super(Qnet, self).__init__()

        s_dim = config.num_tickers
        a_dim = config.num_tickers

        self.config = config

        # First Qnet
        self.qnet1 = nn.Sequential(
            nn.Linear(s_dim + a_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Second Qnet
        self.qnet2 = nn.Sequential(
            nn.Linear(s_dim + a_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def q1(self, state: torch.Tensor, action: torch.Tensor):
        """
        Q-value 1

        state: (batch, num_tickers)
        action: (batch, num_tickers)
        """
        # (batch, num_tickers * 2)
        x = torch.cat([state, action], 1)
        # (batch, 1)
        q = self.qnet1(x)
        return q

    def q2(self, state: torch.Tensor, action: torch.Tensor):
        """
        Q-value 2

        state: (batch, num_tickers, 2)
        action: (batch, num_tickers)
        """
        # (batch, num_tickers * 2)
        x = torch.cat([state, action], 1)
        # (batch, 1)
        q = self.qnet2(x)
        return q
