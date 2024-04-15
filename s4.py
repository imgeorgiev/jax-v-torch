from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import normal, lecun_normal
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.scipy.signal import convolve


def random_SSM(key, N):
    key, a_r, b_r, c_r = jax.random.split(key, 4)
    A = jax.random.uniform(a_r, (N, N))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return key, A, B, C


def discretize(A, B, C, step):
    I = np.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C


def scan_SSM(Ab, Bb, Cb, u, x0):
    """TODO not sure what scan does"""

    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def run_SSM(A, B, C, u):
    L = u.shape[0]  # sequence length
    N = A.shape[0]  # x dimension
    Ab, Bb, Cb = discretize(A, B, C, 1.0 / L)

    todo, todo2 = scan_SSM(Ab, Bb, Cb, u[:, np.newaxis], np.zeros((N,)))
    return todo2


def example_mass(k, b, m):
    A = np.array([[0, 1], [-k / m, -b / m]])
    B = np.array([[0], [1.0 / m]])
    C = np.array([[1.0, 0]])
    return A, B, C


@partial(np.vectorize, signature="()->()")
def example_force(t):
    x = np.sin(10 * t)
    return x * (x > 0.5)


if __name__ == "__main__":
    # For this tutorial, we construct a global JAX rng key
    # but we don't want it when imoprting as a library
    rng = jax.random.PRNGKey(42)

    # SSM
    A, B, C = example_mass(k=40, b=5, m=1)

    inv(A)

    # L samples of u(t).
    L = 100
    step = 1.0 / L
    ks = np.arange(L)
    u = example_force(ks * step)

    # Approximation of y(t).
    y = run_SSM(A, B, C, u)
