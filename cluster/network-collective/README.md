# Network Benchmarks for Collective Operation

## Concept

With respect to collective operations, we are talking about N arrays (input `i_X` and output `o_X`, each resides on rank `X`).

- AllGather:
  - gather data from all nodes, concatanate them, and send back to all nodes.
  - `o_0[K * size] = o_1[K * size] = ... = o_{n-1}[K * size] = i_K`
- AllReduce:
  - perform reduction (e.g. sum) on data from all nodes, and send back to all nodes.
  - `o_0 = o_1 = ... = o_{n-1} = i_0 + i_1 + ... + i_{n-1}`
- ReduceScatter:
  - perform reduction on data from all nodes, and send back each node a portion of the reduced data.
  - `o_K[K * partition size] = i_0 + i_1 + ... + i_{n-1}` where `K` is the rank getting the final result.

## NCCL Test
