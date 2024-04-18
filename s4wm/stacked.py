import jax
import jax.numpy as np
from flax import linen as nn


class SequenceBlock(nn.Module):
    """
    This is a NN wrapper around a sequence layer. It takes a function that maps
    (L,H) -> (L,H) and adds a neural net after the output. Additionally has
    dropout, skip connections and LayerNorm.

    This whole thing is (L, H) -> (L, H)

    NOTE: these are class variables which are shared across all instances!
    NOTE: nn.Module is a python dataclass and automatically sets up __init__
    """

    layer_class: nn.Module
    layer_config: dict
    dropout_rate: float
    model_dim: int
    prenorm: bool = True  # if true applies LayerNorm on input
    glu: bool = True  # gated linear unit
    training: bool = True
    decode: bool = False

    def setup(self):
        self.seq = self.layer_class(**self.layer_config, decode=self.decode)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.model_dim)
        if self.glu:
            self.out2 = nn.Dense(self.model_dim)
        self.drop = nn.Dropout(
            self.dropout_rate,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)
        x = self.drop(nn.gelu(x))
        if self.glu:
            x = self.out(x) * jax.nn.sigmoid(self.out2(x))
        else:
            x = self.out(x)
        x = skip + self.drop(x)
        if not self.prenorm:
            x = self.norm(x)
        return x


class Embedding(nn.Embed):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, x):
        y = nn.Embed(self.num_embeddings, self.features)(x[..., 0])
        return np.where(x > 0, y, 0.0)


class StackedModel(nn.Module):
    """
    Stacks multiple SequenceBlocks together and wraps an encoder and decoder around them.
    This thing now operates on the true data dimension D in the sense
    (L, D) -> (L, H) -> (L, D)
    """

    layer_class: nn.Module
    layer_config: dict
    output_dim: int
    model_dim: int
    n_layers: int
    prenorm: bool = True
    dropout_rate: float = 0.0
    embedding: bool = False  # Use nn.Embed instead of nn.Dense encoder
    classification: bool = False
    training: bool = True
    decode: bool = False  # Probably should be moved into layer_args

    def setup(self):
        if self.embedding:
            self.encoder = Embedding(self.output_dim, self.model_dim)
        else:
            self.encoder = nn.Dense(self.model_dim)
        self.decoder = nn.Dense(self.output_dim)
        self.layers = [
            SequenceBlock(
                layer_class=self.layer_class,
                layer_config=self.layer_config,
                prenorm=self.prenorm,
                model_dim=self.model_dim,
                dropout_rate=self.dropout_rate,
                training=self.training,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        if not self.classification:
            if not self.embedding:
                x = x / 255.0  # Normalize
            if not self.decode:
                x = np.pad(x[:-1], [(1, 0), (0, 0)])
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        if self.classification:
            x = np.mean(x, axis=0)
        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)


# this just makes the StackedModel work with batches (N, L, D) -> (N, L, D)
BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
)
