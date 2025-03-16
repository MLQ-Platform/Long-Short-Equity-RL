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
        self.l1 = nn.Linear(s_dim + a_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, a_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        state: (batch, num_tickers)
        action: (batch, num_tickers) in [-phi, phi]
        """
        # (batch, num_tickers * 2)
        x = torch.cat([state, action], 1)
        # (batch, 400)
        x = self.act(self.l1(x))
        # (batch, 300)
        x = self.act(self.l2(x))
        # (batch, a_dim)
        p = self.out(self.l3(x))

        # (batch, a_dim)
        purturb = self.config.action_scale * p
        # (batch, a_dim)
        purturb_action = (action + purturb).clamp(-1.0, 1.0)
        return purturb_action
