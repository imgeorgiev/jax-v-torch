from flax import linen as nn
import jax


class LSTMRecurrentModel(nn.Module):
    N: int
    L: int
    d_model: int  # model dimension

    def setup(self):
        LSTM = nn.scan(
            nn.OptimizedLSTMCell,
            in_axes=0,
            out_axes=0,
            variable_broadcast="params",
            split_rngs={"params": False},
        )
        dummy_rng = jax.random.PRNGKey(0)  # what is this for???
        self.LSTM = LSTM(name="lstm_cell")
        self.carry = self.LSTM.initialize_carry(dummy_rng, ())
        # NOTE: heaviliy modified


def __call__(self, xs):
    return self.LSTM(self.carry, xs)[1]
