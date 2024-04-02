import os, math
from time import time
from tqdm import tqdm

import jax
import jax.numpy as np
import numpy

import torch

# Now train simple linear regression
xs = numpy.random.normal(size=(100,))
noise = numpy.random.normal(scale=0.1, size=(100,))
ys = xs * 3 - 1 + noise


def model(params, x):
    w = params[0]
    b = params[1]
    return w * x + b


def loss_fn(params, x, y):
    prediction = model(params, x)
    return np.mean((prediction - y) ** 2)


def update(params, x, y, lr=0.01):
    return params - lr * jax.grad(loss_fn)(params, x, y)


@jax.jit
def update_jit(params, x, y, lr=0.01):
    return params - lr * jax.grad(loss_fn)(params, x, y)


print("JAX linear regression")
now = time()
params = np.array([1.0, 0.0])
for _ in tqdm(range(1000)):
    params = update(params, xs, ys)
print("Time taken", time() - now)

now = time()
params = np.array([1.0, 0.0])
for _ in tqdm(range(1000)):
    params = update_jit(params, xs, ys)
print("Time taken with jit", time() - now)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.Tensor([1.0]))
        self.b = torch.nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        return self.w * x + self.b


print("Pytorch linear regression")
model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
_xs = torch.tensor(xs)
_ys = torch.tensor(ys)
now = time()
for _ in tqdm(range(1000)):
    optimizer.zero_grad()
    loss = torch.mean((model(_xs) - _ys) ** 2)
    loss.backward()
    optimizer.step()
print("Time taken", time() - now)

model = torch.compile(Model())
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
_xs = torch.tensor(xs)
_ys = torch.tensor(ys)
now = time()
for _ in tqdm(range(1000)):
    optimizer.zero_grad()
    loss = torch.mean((model(_xs) - _ys) ** 2)
    loss.backward()
    optimizer.step()
print("Time taken jit", time() - now)
