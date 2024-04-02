# follows https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html

import jax
import jax.numpy as np

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
