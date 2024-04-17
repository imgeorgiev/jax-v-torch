from flax import linen as nn


class FeedForwardModel(nn.Module):
    num_units: int
    N: int
    L: int
    decode: bool = False

    def setup(self):
        self.dense = nn.Dense(self.num_units)

    def __call__(self, x):
        """x shape (L, N)"""
        return nn.relu(self.dense(x))
