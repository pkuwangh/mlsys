# Note for D2L-AI

- [Note for D2L-AI](#note-for-d2l-ai)
  - [Concepts](#concepts)
  - [Linear Regression](#linear-regression)

Reference github: https://github.com/d2l-ai/d2l-en/tree/master

## Concepts

`Compute Graph`:
- Break calculation into operators and create an acyclic graph;
- This help automatic gradient calculation using numerical methods.
- Complexity:
  - computational complexity: O(n), similar for forward and backward passes.
  - memory complexity: O(n), intermediate results of forward pass needs to be saved.

## Linear Regression
