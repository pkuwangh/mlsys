#!/usr/bin/env python3

import os
import sys
import time
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel

from redis_store import RedisStore


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_data_parallel():
    rank = torch.distributed.get_rank()
    print(f"Start running DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    # DDP class wraps the model and provides data parallelism by
    # synchronizing gradients across each model replica
    ddp_model = DistributedDataParallel(model, device_ids=[device_id])

    # mean squared error loss
    loss_fn = nn.MSELoss()
    # stochastic gradient descent
    # use only a subset of the data to compute gradients
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # forward pass
    outputs = ddp_model(torch.randn(20, 10))
    # makeup the target labels
    labels = torch.randn(20, 5).to(device_id)
    # calculate the loss
    loss = loss_fn(outputs, labels)
    # backward pass
    loss.backward()
    # update the weights
    optimizer.step()


def demo_all_reduce():
    rank = torch.distributed.get_rank()
    print(f"Start running all-reduce example on rank {rank}.")
    output = torch.tensor([100 + rank]).cuda(0)
    torch.distributed.all_reduce(output)
    torch.distributed.barrier()
    if int(rank) == 0:
        print(f"rank={rank}, output={output}")


if __name__ == "__main__":
    # test env
    MY_RANK = int(os.environ.get("RANK", None))
    MY_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", None))
    MASTER_ADDR = os.environ.get("MASTER_ADDR", None)
    MASTER_PORT = int(os.environ.get("MASTER_PORT", None))
    if any([x is None for x in [MY_RANK, MY_WORLD_SIZE, MASTER_ADDR, MASTER_PORT]]):
        print("RANK or WORLD_SIZE or MASTER_ADDR or MASTER_PORT is not set.")
        sys.exit(1)

    print("Initializing process group.")
    # initialize the distributed environment
    # create a process group and set the communication backend to be NCCL
    # assign each process a unique rank and initialize the network connections
    if MASTER_PORT == 6379:
        redis_store = RedisStore(MASTER_ADDR, MASTER_PORT, MY_RANK == 0)
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=MY_WORLD_SIZE,
            rank=MY_RANK,
            store=redis_store,
        )
    else:
        torch.distributed.init_process_group(backend="nccl")
    print("init_process_group done.", flush=True)
    redis_store.dump_keys()
    time.sleep(3)

    print("\nEntering barrier1...", flush=True)
    torch.distributed.barrier()
    print("\nPassing barrier1...", flush=True)
    redis_store.dump_keys()
    time.sleep(3)

    print("\nCreating group...", flush=True)
    group = torch.distributed.new_group(range(MY_WORLD_SIZE), backend="gloo")
    redis_store.dump_keys()
    time.sleep(3)

    # Run a single training step
    # demo_data_parallel()

    print("\nEntering barrier2...", flush=True)
    torch.distributed.barrier()
    print("\nPassing barrier2...", flush=True)

    # Run a multi-gpu all reduce
    # demo_all_reduce()
    # torch.distributed.barrier()

    torch.distributed.destroy_process_group()
