import pickle
from rlweight.ddpg.trainer import ReplayBuffer


def save_buffer(buffer: ReplayBuffer, filename="buffer.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(buffer, f)


def load_buffer(filename="buffer.pkl"):
    with open(filename, "rb") as f:
        buffer = pickle.load(f)
    return buffer
