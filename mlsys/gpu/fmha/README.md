# Attention Kernels

## Basics

```python
O = Dropout(Softmax(Mask(Q * K.T))) * V
```

Key intuition is

- For each sequence of N tokens, attention matrix is NxN, which again is per sequence.
- The key is to avoid reading/writing the whole attention matrix repeatedly.

So we will do tiling, which gives a BN x BN tile of S; the challenge is

- How to run `softmax` when only part of the row is available.
- In backward pass, the intermediate attention scores are needed, which we can do re-compute.

The key idea for the softmax problem is, you can run softmax on partial vectors and re-scale.

```python
softmax([A1, A2]) * [V1, V2].T = alpha * softmax(A1) * V1 + beta * softmax(A2) * V2
```

Note both `alpha` & `beta` still depend on both `A1` & `A2`, so some cross-block traffic is still needed, but they are scaler values.

### Flash-Attention

```python
# formulation
p_i = exp(l_i) / sum(exp(l_i))

# to make it numatically stable
p_i = exp(l_i - m) / sum(exp(l_i - m))

# to avoid the sum reduction
sum(exp(l_i - m_new)) = exp(m - m_new) * sum(exp(l_i - m))
```
