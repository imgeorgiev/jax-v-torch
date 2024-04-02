# JAX vs PyTorch

A repository I'm using to compare JAX to PyTorch across

- ease of use (1-5)
- implementation speed
- training speed
- breadth of community support

# Qualitative

All measures on a completel arbitrary scale of 1-5
         
| Model | PT Ease of use | PT Implementation speed | PT Support | JAX Ease of use | JAX Implementation speed | JAX Support |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Linear regression | 5 | 5 | 5 | 5 | 5 | 5 |


# Training speeeds

## Apple Silicon

Tested on M1 Max with

### Raw results (in seconds)

| Model | PyTorch | PyTorch compiled | JAX | JAX compiled |
| ---- | ---- | ---- | ---- | ---- |
| Linear Regression | 0.0943 | 1.1649 | 7.3239 | 0.4568 |

### Relative results (unit of time)

| Model | PyTorch | PyTorch compiled | JAX | JAX compiled |
| ---- | ---- | ---- | ---- | ---- |
| Linear Regression | 1.0 | 12.35 | 77.6660 | 4.8441 |