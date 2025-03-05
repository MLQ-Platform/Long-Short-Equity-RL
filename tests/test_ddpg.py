import torch

from rlweight.ddpg.model import Actor
from rlweight.ddpg.model import Critic
from rlweight.ddpg.model import ModelConfig


def test_actor():
    """
    DDPG Actor feed-forward test
    """
    NUM_TICKERS = 29

    config = ModelConfig(num_tickers=NUM_TICKERS)
    actor = Actor(config)

    state = torch.randn(1, NUM_TICKERS, 2)
    action = actor(state)

    assert action.shape == (1, 1)


def test_critic():
    """
    DDPG Critic feed-forward test
    """
    NUM_TICKERS = 29

    config = ModelConfig(num_tickers=NUM_TICKERS)
    critic = Critic(config)

    state = torch.randn(1, NUM_TICKERS, 2)
    action = torch.randn(1, 1)

    q_value = critic(state, action)

    assert q_value.shape == (1, 1)
