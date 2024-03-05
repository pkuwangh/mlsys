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

In a BCM cluster, first initialize LD_LIBRARY_PATH:

```bash
source ../source_me_bcm.sh
echo $LD_LIBRARY_PATH
```

The expected final command should look like

```bash
mpirun --mca btl tcp,self -np 2 --host s-s126-bot-dvt1c-ks-1174,s-s126-bot-dvt1c-ks-1175 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH sendrecv_perf -t1 -g1 -b16 -e2g -f2 -c0
```

The runner script can help

```bash
# provide comma-separated hosts directly
./run-nccl-tests-baremetal-mpi.py -r s-s126-bot-dvt1c-ks-1174,s-s126-bot-dvt1c-ks-1175 -p /home/nvidia/haowan/nccl_test --test all_reduce --test sendrecv

# use a host file - at most 2 levels of hierarchy
./run-nccl-tests-baremetal-mpi.py -f ./hosts/superpowers.yml -d l1_01 -p /home/nvidia/haowan/nccl_test --test all_reduce --test sendrecv
```
