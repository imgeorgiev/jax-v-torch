from flax import linen as nn
import jax.numpy as np


class FeedForwardModel(nn.Module):
    N: int
    l_max: int
    model_dim: int = 128
    decode: bool = False

    def setup(self):
        self.dense = nn.Dense(self.model_dim)

    def __call__(self, x):
        """x shape (L, N)"""
        return nn.relu(self.dense(x))


class FeedForwardAutoregressive(nn.Module):
    N: int
    l_max: int
    model_dim: int = 128
    decode: bool = False

    def setup(self):
        self.dense = nn.Dense(self.model_dim)

    def __call__(self, x):
        """x shape (L, N)"""
        xs = []
        x = x[0]
        for l in range(self.l_max):
            x = nn.relu(self.dense(x))
            xs.append(x)
        return np.array(xs)
