import torch
import torch.nn as nn
from rlweight.bcq.model.config import ModelConfig


class Perturbation(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Perturbation, self).__init__()

        s_dim = config.num_tickers
        a_dim = config.num_tickers

        self.config = config
        self.act = nn.ReLU()
        self.out = nn.Tanh()

        self.l1 = nn.Linear(s_dim + a_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, a_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        state: (batch, num_tickers)
        action: (batch, num_tickers) in [-phi, phi]
        """
        # (batch, num_tickers * 2)
        x = torch.cat([state, action], 1)
        # (batch, 128)
        x = self.act(self.l1(x))
        # (batch, 64)
        x = self.act(self.l2(x))
        # (batch, a_dim)
        p = self.out(self.l3(x))

        # (batch, a_dim)
        perturb = self.config.perturb_scale * p
        # (batch, a_dim)
        perturb_action = (action + perturb).clamp(-1.0, 1.0)
        return perturb_action
