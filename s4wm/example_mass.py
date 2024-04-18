from functools import partial
import jax
import jax.numpy as np

from .ssm import run_SSM


@partial(np.vectorize, signature="()->()")
def example_force(t):
    x = np.sin(10 * t)
    return x * (x > 0.5)


if __name__ == "__main__":
    """
    This example is a minimal example of prediction via SSMs
    We define a 1-dim state 1-dm action spring damper system where we input
    a force u and observe the position x. The observed position is given by the SSSM

    In this example we are running vanilla SSMs which do 1d sequence prediction
    (L,) -> (L,)
    """

    # define SSM matrices
    k = 40
    b = 5
    m = 1
    A = np.array([[0, 1], [-k / m, -b / m]])
    B = np.array([[0], [1.0 / m]])
    C = np.array([[1.0, 0]])

    # L samples of u(t).
    L = 100  # sequence length
    step = 1.0 / L
    ks = np.arange(L)
    u = example_force(ks * step)

    # Approximation of y(t).
    y = run_SSM(A, B, C, u)

    # Plotting ---
    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_context("paper")
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_title("Force $u_k$")
    ax2.set_title("Position $y_k$")
    ax1.set_xticks([], [])
    ax2.set_xticks([], [])
    ax1.plot(ks, u)
    ax2.plot(ks, y)
    plt.savefig("example_mass.pdf")
