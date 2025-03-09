import torch
import torch.nn as nn
from rlweight.ddpg.model.config import ModelConfig


class Critic(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Critic, self).__init__()
        self.config = config

        self.state_fc = nn.Sequential(
            nn.Linear(config.num_tickers, 64),
            nn.ReLU(),
        )

        self.action_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
        )

        self.fc_module = nn.Sequential(
            nn.Linear(64 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        state: (batch, num_tickers, 2)
        action: (batch, num_tickers)
        """
        # (batch, 64)
        state_emb = self.state_fc(state)
        # (batch, 64)
        action_emb = self.action_fc(action)
        # (batch, 64 + 64)
        q_input = torch.cat([state_emb, action_emb], dim=1)
        # (batch, 1)
        q_value = self.fc_module(q_input)
        return q_value
