import torch
import torch.nn as nn
import numpy as np
from typing import List
from rlweight.ddpg.model.actor import Actor


class EnsembleActor(nn.Module):
    """
    DDPG Actor based Ensemble Network
    """

    def __init__(self, actors: List[Actor]):
        super(EnsembleActor, self).__init__()

        # Create ensemble of actors
        self.actors = nn.ModuleList(actors)

    def forward(self, state: torch.Tensor):
        """
        state: (num_batch, num_tickers)
        """
        # (num_actors, num_batch, num_actions)
        actions = torch.stack([actor(state) for actor in self.actors], dim=0)
        # (num_batch, num_actions)
        avg_action = torch.mean(actions, dim=0)
        return avg_action

    def from_numpy(self, state: np.ndarray) -> np.ndarray:
        """
        state: (num_tickers,)
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        return self(state).detach().numpy().squeeze(0)
