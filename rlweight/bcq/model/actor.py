import torch
import torch.nn as nn
from rlweight.bcq.model.config import ModelConfig


class VAE(nn.Module):
    """
    Batch Constrained Action Generator
    """

    def __init__(self, config: ModelConfig):
        super(VAE, self).__init__()

        s_dim = 2 * config.num_tickers
        a_dim = config.num_tickers
        z_dim = a_dim * 2

        self.z_dim = z_dim
        self.act = nn.ReLU()
        self.out = nn.Tanh()

        # Layers for Encoder
        self.e1 = nn.Linear(s_dim + a_dim, 750)
        self.e2 = nn.Linear(750, 750)
        self.mu = nn.Linear(750, z_dim)
        self.log_std = nn.Linear(750, z_dim)

        # Layers for Decoder
        self.d1 = nn.Linear(s_dim + z_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, a_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        state: (batch, num_tickers, 2)
        action: (batch, num_tickers)
        """

        # (batch, num_tickers, 2) -> (batch, num_tickers * 2)
        state = state.reshape(state.size(0), -1)

        z, mean, std = self.encoder(state, action)
        a = self.decoder(state, z)
        return a, mean, std

    def encoder(self, state: torch.Tensor, action: torch.Tensor):
        # (batch, num_tickers, 2) -> (batch, num_tickers * 2)
        state = state.reshape(state.size(0), -1)
        # (batch, num_tickers * 3)
        x = torch.cat([state, action], 1)
        # (batch, 750)
        x = self.act(self.e1(x))
        # (batch, 750)
        x = self.act(self.e2(x))
        # (batch, a_dim * 2)
        mean = self.mu(x)
        # (batch, a_dim * 2)
        log_std = self.log_std(x)
        # (batch, a_dim * 2)
        log_std = self.log_std(x).clamp(-4, 15)
        # (batch, a_dim * 2)
        std = torch.exp(log_std)
        # (batch, a_dim * 2)
        z = mean + std * torch.randn_like(std)
        return z, mean, std

    def decoder(self, state: torch.Tensor, z: torch.Tensor = None):
        # (batch, num_tickers, 2) -> (batch, num_tickers * 2)
        state = state.reshape(state.size(0), -1)
        # (batch, a_dim * 2)
        z_shape = (state.shape[0], self.z_dim)
        # (batch, a_dim * 2)
        z = torch.randn(z_shape).clamp(-0.5, 0.5) if z is None else z
        # (batch, num_tickers * 4)
        x = torch.cat([state, z], 1)
        # (batch, 750)
        x = self.act(self.d1(x))
        # (batch, 750)
        x = self.act(self.d2(x))
        # (batch, a_dim)
        action = self.out(self.d3(x))
        return action
