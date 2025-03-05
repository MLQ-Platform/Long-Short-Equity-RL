import torch

from rlweight.bcq.model import VAE
from rlweight.bcq.model import Qnet
from rlweight.bcq.model import Perturbation
from rlweight.bcq.model import ModelConfig


def test_vae():
    """
    VAE feed-forward test
    """
    NUM_TICKERS = 29

    config = ModelConfig(num_tickers=NUM_TICKERS)
    vae = VAE(config)

    state = torch.randn(1, NUM_TICKERS, 2)
    action = torch.randn(1, NUM_TICKERS)

    z, mean, std = vae.encoder(state, action)
    action = vae.decoder(state, z)

    assert z.shape == (1, NUM_TICKERS * 2)
    assert mean.shape == (1, NUM_TICKERS * 2)
    assert std.shape == (1, NUM_TICKERS * 2)
    assert action.shape == (1, NUM_TICKERS)


def test_qnet():
    """
    Qnet feed-forward test
    """
    NUM_TICKERS = 29

    config = ModelConfig(num_tickers=NUM_TICKERS)
    qnet = Qnet(config)

    state = torch.randn(1, NUM_TICKERS, 2)
    action = torch.randn(1, NUM_TICKERS)

    q1 = qnet.q1(state, action)
    q2 = qnet.q2(state, action)

    assert q1.shape == (1, 1)
    assert q2.shape == (1, 1)


def test_perturbation():
    """
    Perturbation feed-forward test
    """

    NUM_TICKERS = 29

    config = ModelConfig(num_tickers=NUM_TICKERS)
    perturbation = Perturbation(config)

    state = torch.randn(1, NUM_TICKERS, 2)
    action = torch.randn(1, NUM_TICKERS)

    purturb_action = perturbation(state, action)

    assert purturb_action.shape == (1, NUM_TICKERS)
