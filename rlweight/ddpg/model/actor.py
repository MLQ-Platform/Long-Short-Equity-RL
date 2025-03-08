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
            nn.Linear(64, config.num_tickers),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor):
        """
        state: (num_batch, num_tickers, 2)
        """

        act = self.fc_module(state)

        # Scaling
        act = act / torch.abs(act).sum(dim=1, keepdim=True)
        return act
