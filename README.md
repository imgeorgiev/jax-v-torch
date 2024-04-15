# JAX vs PyTorch

A repository I'm using to compare JAX to PyTorch across

- ease of use
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

Tested on M1 Max. Median iteration time.

### Raw results (in seconds)

| Model | PyTorch | PyTorch compiled | JAX | JAX compiled |
| ---- | ---- | ---- | ---- | ---- |
| Linear Regression | 0.00009 | 0.00013 | 0.0067 | 0.00545 |

### Relative results (unit of time)

| Model | PyTorch | PyTorch compiled | JAX | JAX compiled |
| ---- | ---- | ---- | ---- | ---- |
| Linear Regression | 1.0 | 1.44 | 74.44 | 60.55z |