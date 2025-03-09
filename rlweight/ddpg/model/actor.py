import torch
import torch.nn as nn
from rlweight.ddpg.model.config import ModelConfig


class Actor(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Actor, self).__init__()

        self.config = config

        # Fully Connected Layer
        self.fc_module = nn.Sequential(
            nn.Linear(config.num_tickers, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor):
        """
        state: (num_batch, num_tickers)
        """
        act = self.fc_module(state)
        return act
