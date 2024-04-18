from time import time
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh, inv, matrix_power
from .ssm import random_SSM, discretize
from .utils import scan_SSM, causal_convolution, log_step_initializer, cloneLayer

from .math import cauchy_dot, cauchy

if __name__ == "__main__":
    # For this tutorial, construct a global JAX rng key
    # But we don't want it when importing as a library
    rng = jax.random.PRNGKey(1)


# S4 models are the evolution of SSMs to make them feasible for deep learning.
# S4s introduce two main changes
# 1) Special A matrix (HiPPO) for long-range dependencies
# 2) A computational optimization for the recursive application of A
# 2.1) SSM Generating function


# HIPPO matrix enables sustainable matrix initialization for long sequences
def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


# For the speedup we are improving the K convolution
def K_conv(Ab, Bb, Cb, L):
    return np.array([(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)])


# 2.1) Instead of taking matrix powers like in above, we can do an alternation
#       where we only need to take a single inverse which is much more simple.
#       This is done with truncated SSM generating functions at nodes $z$.
#       $K_L(z; A, B, C) \in \C := \sum_{i=0}^{L-1} C A^i B z^i$
#       Also called a z-transform. It is effectively a compression of the original
#       filter in complex number space in frequency space.
#       We can later recover this into discret time space using a z-transform at the roots of unity


# simple generating function
def K_gen_simple(Ab, Bb, Cb, L):
    K = K_conv(Ab, Bb, Cb, L)
    return lambda z: np.sum(K * (z * np.arange(L)))


# A more efficient version of this is the inverse version
# TODO expand on the map
def K_gen_inverse(Ab, Bb, Cb, L):
    I = np.eye(Ab.shape[0])
    Ab_L = matrix_power(Ab, L)
    Ct = Cb @ (I - Ab_L)
    return lambda z: (Ct.conj() @ inv(I - Ab * z) @ Bb).reshape()


# We use this to recover a convolution from a generating function
def conv_from_gen(gen, L):
    # First get roots of unity
    Omega_L = np.exp((-2j * np.pi) * np.arange(L) / L)
    eval_at_roots = jax.vmap(gen)(Omega_L)  # vmap for vectorization

    # Now do a inverse FFT to recover the convolution
    out = np.fft.ifft(eval_at_roots, L).reshape(L)
    return out.real


# Now let's compare speeds!
if __name__ == "__main__":
    print("Testing K conv generating functions")
    L = 16
    N = 4
    iters = 100
    ssm = random_SSM(rng, N)
    ssm = discretize(*ssm, 1 / L)

    # first do the simple K convolution computation
    now = time()
    for _ in range(iters):
        a = K_conv(*ssm, L)
    took = time() - now
    print(f"K_conv took {took*1e3/iters:.2f}ms")

    # Now do the simple generating function
    now = time()
    for _ in range(iters):
        b = conv_from_gen(K_gen_simple(*ssm, L), L)
    took = time() - now
    print(f"K_gen_simple took {took*1e3/iters:.2f}ms")

    # Now do the simple generating function
    now = time()
    for _ in range(iters):
        c = conv_from_gen(K_gen_inverse(*ssm, L), L)
    took = time() - now
    print(f"K_gen_inverse took {took*1e3/iters:.2f}ms")

    assert np.allclose(a, c)


# 2.2) SSMs with diagonal A make it way easier to compute the inverse
#       by using a Cauchy kernel


# 2.3) In addition to the diagonal, we also itnroduce a low-rank component P, Q \in C^{N}
#       Called a Diagonal Plus Low-Rank (DPLR) matrix


def random_DPLR(rng, N):
    l_r, p_r, q_r, b_r, c_r = jax.random.split(rng, 5)
    Lambda = jax.random.uniform(l_r, (N,))
    P = jax.random.uniform(p_r, (N,))
    Q = jax.random.uniform(q_r, (N,))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return Lambda, P, Q, B, C


def kernel_DPLR(Lambda, P, Q, B, C, step, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = np.exp((-2j * np.pi) * (np.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = np.fft.ifft(atRoots, L).reshape(L)
    return out.real


if __name__ == "__main__":
    # Now test the new K kernel computation method
    lambd, P, Q, B, C = random_DPLR(rng, N)
    step = 1 / L

    now = time()
    for _ in range(iters):
        d = kernel_DPLR(lambd, P, P, B, C, step, L)
    took = time() - now
    print(f"kernel_DPLR took {took*1e3/iters:.2f}ms")

# Now we need to fit these DPLR into the HiPPO concept.
# First we make a Normal Plus Low Rank HiPPO matrix and then convert it to DPLR


def make_NPLR_HiPPO(N):
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return nhippo, P, B


def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    # Check skew symmetry
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    # assert np.allclose(Lambda_real, S_diag, atol=1e-3) # TODO shouldn't this be here?

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V


def DPLR_to_SSM(Lambda, P, Q, B, C, step, L):
    """Converts a DPLR matrix to discrete SSM matrices
    DPLR matrices are used when doing convolutions with S4s.
    Discrete SSM matrices are used for autoregression"""
    # Convert parameters to matrices
    B = B[:, np.newaxis]
    Ct = C[np.newaxis, :]

    N = Lambda.shape[0]
    A = np.diag(Lambda) - P[:, np.newaxis] @ Q[:, np.newaxis].conj().T
    I = np.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = np.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()


if __name__ == "__main__":
    print("Sanity check for DPLR matrix generation")
    lambd, P, B, V = make_DPLR_HiPPO(N)
    C = normal(dtype=np.complex64)(rng, (N,))
    K = kernel_DPLR(lambd, P, P, B, C, step, L)

    Ab, Bb, Cb = DPLR_to_SSM(lambd, P, P, B, C, step, L)
    K2 = K_conv(Ab, Bb, Cb, L)
    assert np.allclose(K.real, K2.real)


class S4Layer(nn.Module):
    """Similar to the SSM alyer but we compute K differently and We learn C tidle
    instead of C to avoid computing powers of A"""

    N: int
    l_max: int
    decode: bool = False

    # Special parameters with multiplicative factor on lr and no weight decay (handled by main train script)
    lr = {
        "Lambda_re": 0.1,
        "Lambda_im": 0.1,
        "P": 0.1,
        "B": 0.1,
        "log_step": 0.1,
    }

    def setup(self):
        # Learned Parameters (C is complex!)
        init_A_re, init_A_im, init_P, init_B = hippo_initializer(self.N)
        self.Lambda_re = self.param("Lambda_re", init_A_re, (self.N,))
        self.Lambda_im = self.param("Lambda_im", init_A_im, (self.N,))
        # Ensure the real part of Lambda is negative
        # (described in the SaShiMi follow-up to S4)
        self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        self.P = self.param("P", init_P, (self.N,))
        self.B = self.param("B", init_B, (self.N,))
        # C should be init as standard normal
        # This doesn't work due to how JAX handles complex optimizers https://github.com/deepmind/optax/issues/196
        # self.C = self.param("C", normal(stddev=1.0, dtype=np.complex64), (self.N,))
        self.C = self.param("C", normal(stddev=0.5**0.5), (self.N, 2))
        self.C = self.C[..., 0] + 1j * self.C[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = np.exp(self.param("log_step", log_step_initializer(), (1,)))

        if not self.decode:
            # CNN mode, compute kernel.
            self.K = kernel_DPLR(
                self.Lambda,
                self.P,
                self.P,
                self.B,
                self.C,
                self.step,
                self.l_max,
            )

        else:
            # RNN mode, discretize

            # Flax trick to cache discrete form during decoding.
            def init_discrete():
                return DPLR_to_SSM(
                    self.Lambda,
                    self.P,
                    self.P,
                    self.B,
                    self.C,
                    self.step,
                    self.l_max,
                )

            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

            # RNN Cache
            self.x_k_1 = self.variable(
                "cache", "cache_x_k", np.zeros, (self.N,), np.complex64
            )

    def __call__(self, u):
        # This is identical to SSM Layer
        if not self.decode:
            # CNN Mode
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


S4Layer = cloneLayer(S4Layer)


def init_recurrence(model, params, init_x, rng):
    """We first precompute the discretizaed version of the RNN for each S4
    layer. We do this to the 'prime' collection of variables"""
    variables = model.init(rng, init_x)
    vars = {
        "params": params,
        "cache": variables["cache"].unfreeze(),
        "prime": variables["prime"].unfreeze(),
    }
    print("Priming")
    _, prime_vars = model.apply(vars, init_x, mutable=["prime"])
    return vars["params"], prime_vars["prime"], vars["cache"]


# Factory for constant initializer in Flax
def init(x):
    def _init(key, shape):
        assert shape == x.shape
        return x

    return _init


def hippo_initializer(N):
    Lambda, P, B, _ = make_DPLR_HiPPO(N)
    return init(Lambda.real), init(Lambda.imag), init(P), init(B)
