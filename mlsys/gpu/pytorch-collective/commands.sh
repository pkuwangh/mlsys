#!/bin/bash

# 1 node
torchrun --nproc_per_node=4 test-torchrun-basics.py
torchrun --nproc_per_node=4 test-pytorch-distributed.py

# 2 nodes
# set node0 to node-0's IP addr
torchrun --nnodes=2 --nproc_per_node=4 --node-rank=0 --master_addr=$node0 --master_port=12341 test-torchrun-basics.py
torchrun --nnodes=2 --nproc_per_node=4 --node-rank=1 --master_addr=$node0 --master_port=12341 test-torchrun-basics.py

# set NCCL_SOCKET_IFNAME & GLOO_SOCKET_IFNAME
torchrun --nnodes=2 --nproc_per_node=4 --node-rank=0 --master_addr=$node0 --master_port=12341 test-pytorch-distributed.py
torchrun --nnodes=2 --nproc_per_node=4 --node-rank=1 --master_addr=$node0 --master_port=12341 test-pytorch-distributed.py
