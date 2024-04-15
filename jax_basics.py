# follows https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html

import jax
import jax.numpy as np
import numpy
from time import time

x = np.arange(10).astype(np.float32)
print(x)
print(x.dtype)
print(type(x))


def sum_of_squares(x):
    return np.sum(x**2)


# We can think of JAX as numpy with gradients

grad_fn = jax.grad(sum_of_squares)
print("y", sum_of_squares(x))
print("grad", grad_fn(x))

# this is pretty similar to grad(f)(x) where f is the function we are trying to take the gradient of
# This is much closer to how we think of gradients and the underlying math


# We can also do gradient computation for loss functions with a 2nd variable
def sum_sq_error(x, y):
    return np.sum((x - y) ** 2)


grad_fn = jax.grad(sum_sq_error)
y = x + 0.1
print("y", sum_sq_error(x, y))
print("grad", grad_fn(x, y))

# we can also get both at the same time
loss, grad = jax.value_and_grad(sum_sq_error)(x, y)


# The core thing we need to remember is that JAX is a functional programming library
#   and as such all functions need to return something and can't modify their inputs
def in_place_modify(x):  #  BAD!
    x[0] = 123
    return None


def jax_in_place_modify(x):  # GOOD!
    return x.at[0].set(123)


x = np.arange(10)
try:
    in_place_modify(x)
except Exception as e:
    print("In place modify failed")

jax_in_place_modify(x)

###3333##############################
# Vectorization with vmap
print("\nVectorization")


@jax.jit
def convolve(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(np.dot(x[i - 1 : i + 2], w))
    return np.array(output)


x = np.arange(5)
w = np.array([2.0, 3.0, 4.0])
xs = np.stack([x, x])
ws = np.stack([w, w])

# manual vectorization/batching
convolve(xs[0], ws[0])  # warm start
now = time()
output = []
for i in range(xs.shape[0]):
    output.append(convolve(xs[i], ws[i]))
output = np.stack(output)
print("Manual batching", time() - now)

# automatic vectorization/batching
now = time()
auto_batch_convolve = jax.vmap(convolve)
auto_batch_convolve(xs, ws)
now = time()
auto_batch_convolve(xs, ws)
print("Auto batching", time() - now)


#################################
print("\nAdvanced differentiation")

# jax makes it easy to do higher order gradients
f = lambda x: x**3 + 2 * x**2 - 3 * x + 1

dfdx = jax.grad(f)
d2fdx = jax.grad(dfdx)
d3fdx = jax.grad(d2fdx)
d4fdx = jax.grad(d3fdx)

print("1st order", dfdx(1.0))
print("2nd order", d2fdx(1.0))
print("3rd order", d3fdx(1.0))
print("4th order", d4fdx(1.0))


# we can also get hessians
def hessian(f):
    return jax.jacfwd(jax.grad(f))  # forward mode
    # return jax.jacrev(jax.grad(f)) # reverse mode


def f(x):
    return np.dot(x, x)


hess = hessian(f)(np.array([1.0, 2.0, 3.0]))
print("Hessian")
print(hess)

# we can stop gradients via this example from RL
s_tm1 = np.array([1.0, 2.0, -1.0])
r_t = np.array(1.0)
s_t = np.array([2.0, 1.0, 0.0])

value_fn = lambda theta, state: np.dot(theta, state)


def td_loss(theta, s_tm1, r_t, s_t):
    v_tm1 = value_fn(theta, s_tm1)
    target = r_t + value_fn(theta, s_t)
    return -0.5 * ((jax.lax.stop_gradient(target) - v_tm1) ** 2)


# computing gradients per data sample efficiently
perex_grads = jax.jit(jax.vmap(jax.grad(td_loss), in_axes=(None, 0, 0, 0)))
# 1. we take grad of the loss function
# 2. We vectorize it over all arguments except the parameters (since it's a single model)
# 3. We jit compile it for efficiency

# Test it:
theta = np.array([0.1, -0.1, 0.0])
batched_s_tm1 = np.stack([s_tm1, s_tm1])
batched_r_t = np.stack([r_t, r_t])
batched_s_t = np.stack([s_t, s_t])

perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t)


#######################################
print("\nRandom numbers")
# JAX random numbers have to be
# 1. instantiated with a key
# 2. Consumed immediately by some random process
# 3. The key is updated and passed to the next random process

from jax import random  # different from numpy!

key = random.key(42)
print("old key", key)

print("using the same key", random.normal(key), random.normal(key))
key, subkey = random.split(key)
print("subkey consumed immediately", random.normal(subkey))

# can also generate multiple random numbers
key, *forty_two_subkeys = random.split(key, num=43)

#####################################
print("\nPytrees")
# nested dict like data structures that make things easier
# example of training an MLP


def forward(params, x):
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu(x @ layer["weights"] + layer["biases"])
    return x @ last["weights"] + last["biases"]


def loss_fn(params, x, y):
    return np.mean((forward(params, x) - y) ** 2)


LEARNING_RATE = 0.0001


@jax.jit
def update(params, x, y):

    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    # Note that `grads` is a pytree with the same structure as `params`.
    # `jax.grad` is one of the many JAX functions that has
    # built-in support for pytrees.

    # This is handy, because we can apply the SGD update using tree utils:
    return loss, jax.tree_map(lambda p, g: p - LEARNING_RATE * g, params, grads)


def init_mlp_params(layer_widths):
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            dict(
                weights=numpy.random.normal(size=(n_in, n_out)) * np.sqrt(2 / n_in),
                biases=numpy.ones(shape=(n_out,)),
            )
        )
    return params


params = init_mlp_params([1, 128, 128, 1])

print("NN init parameter shapes")
print(jax.tree_map(lambda x: x.shape, params))

xs = numpy.random.normal(size=(128, 1))
ys = xs**2

for _ in range(100):
    loss, params = update(params, xs, ys)
    print(f"Loss: {loss:.3f}", end="\r")
print(f"final loss = {loss:.3f}")


print("Example of a tree map debugging")
import collections
from typing import NamedTuple, Iterable

ATuple = collections.namedtuple("ATuple", ("name"))

tree = [1, {"k1": 2, "k2": (3, 4)}, ATuple("foo")]
flattened, _ = jax.tree_util.tree_flatten_with_path(tree)
for key_path, value in flattened:
    print(f"Value of tree{jax.tree_util.keystr(key_path)}: {value}")


print("Custom pytrees")

# We can't simply define arbitrary classes, put them in pytrees and expect them
# to fit in a tree-like structure. We have to explicitly define it.


class MyContainer:
    def __init__(self, name: str, a, b, c):
        self.name = name
        self.a = a
        self.b = b
        self.c = c


def flatten(container: MyContainer) -> tuple[Iterable[int], str]:
    """Return amn iterable over container contetnts, and aux data"""
    flat_contetnts = [container.a, container.b, container.c]
    aux_data = container.name
    return flat_contetnts, aux_data


def unflatten(aux_data: str, flat_contents) -> MyContainer:
    return MyContainer(aux_data, *flat_contents)


# now register with JAX
jax.tree_util.register_pytree_node(MyContainer, flatten, unflatten)
output = jax.tree_util.tree_leaves(
    [MyContainer("foo", 1, 2, 3), MyContainer("bar", 4, 5, 6)]
)
print("Custom nodes", output)


# we can also do similar things with NamedTupled with the caveat
# that all content of the named tuple appear in the flat tree
class MyNamedTuple(NamedTuple):
    name: str
    a: int
    b: int
    c: int


tree = [MyNamedTuple("foo", 1, 2, 3), MyNamedTuple("bar", 4, 5, 6)]
output = jax.tree_util.tree_leaves(tree)
print("NamedTuple", output)


##########################################
print("\nParallel evaluation")
# JAX is built for parallel execution in a single-program multiple-data (SPMD) style
# e.g. forward pass on NN is run on different input data (e.g. batch) in parallel
# on different devices (e.g. several TPUs)
# Conceptually this is pretty similar to vectorization, but it's more general
# We can use jax.pmap to parallelize a function over multiple devices

# simplest way of doing multi-device parallelization
n_devices = jax.local_device_count()
print(f"Found {n_devices} devices")
xs = np.arange(5 * n_devices).reshape(-1, 5)
ws = np.stack([w] * n_devices)
auto_batch_convolve = jax.pmap(convolve)
auto_batch_convolve(xs, ws)

# I really like the vmap and pmap interoperatability. It just makes scaling easier!


##########################################
print("\nStatefuless in JAX")
# JAX tries to be stateless but sometimes in ML we need states
# e.g. model parameters, optimizers, some layers like BatchNorm


class Counter:
    """A simple counter."""

    def __init__(self):
        self.n = 0

    def count(self) -> int:
        """Increments the counter and returns the new value."""
        self.n += 1
        return self.n

    def reset(self):
        """Resets the counter to zero."""
        self.n = 0


counter = Counter()
fast_count = jax.jit(counter.count)

print("Incorrect counting")
for _ in range(3):
    print(fast_count())

# Now the correct way
CounterState = int


class CounterV2:

    def count(self, n: CounterState) -> tuple[int, CounterState]:
        # You could just return n+1, but here we separate its role as
        # the output and as the counter state for didactic purposes.
        return n + 1, n + 1

    def reset(self) -> CounterState:
        return 0


counter = CounterV2()
state = counter.reset()

print("Correct counting")
for _ in range(3):
    value, state = counter.count(state)
    print(value)


# Generally JAX avoids classes and prefers functions which have states
# Linear regression example:
class Params(NamedTuple):
    weight: np.ndarray
    bias: np.ndarray


def init(rng) -> Params:
    """Returns the initial model params."""
    weights_key, bias_key = jax.random.split(rng)
    weight = jax.random.normal(weights_key, ())
    bias = jax.random.normal(bias_key, ())
    return Params(weight, bias)


def loss(params: Params, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes the least squares error of the model's predictions on x against y."""
    pred = params.weight * x + params.bias
    return np.mean((pred - y) ** 2)


LEARNING_RATE = 0.005


@jax.jit
def update(params: Params, x: np.ndarray, y: np.ndarray) -> Params:
    """Performs one SGD update step on params using the given data.
    Note how we manually pipe parameters in and out of update"""
    grad = jax.grad(loss)(params, x, y)

    # If we were using Adam or another stateful optimizer,
    # we would also do something like
    # ```
    # updates, new_optimizer_state = optimizer(grad, optimizer_state)
    # ```
    # and then use `updates` instead of `grad` to actually update the params.
    # (And we'd include `new_optimizer_state` in the output, naturally.)

    new_params = jax.tree_map(lambda param, g: param - g * LEARNING_RATE, params, grad)

    return new_params


rng = jax.random.key(42)

# Generate true data from y = w*x + b + noise
true_w, true_b = 2, -1
x_rng, noise_rng = jax.random.split(rng)
xs = jax.random.normal(x_rng, (128, 1))
noise = jax.random.normal(noise_rng, (128, 1)) * 0.5
ys = xs * true_w + true_b + noise

# Fit regression
params = init(rng)
for _ in range(1000):
    params = update(params, xs, ys)
