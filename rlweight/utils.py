import uuid
import pickle
from rlweight.ddpg.trainer import ReplayBuffer


def save_buffer(buffer: ReplayBuffer, filename="buffer.pkl"):
    """
    Save a buffer to a file.
    """
    with open(filename, "wb") as f:
        pickle.dump(buffer, f)


def load_buffer(filename="buffer.pkl"):
    """
    Load a buffer from a file.
    """
    with open(filename, "rb") as f:
        buffer = pickle.load(f)
    return buffer


def generate_uuid():
    """
    Generate a random 8-character UUID.
    """
    return str(uuid.uuid4())[:8]
