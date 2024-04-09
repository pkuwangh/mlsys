#!/usr/bin/env python3

import os
import socket
import sys
import time
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
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
        print(f"rank={rank}, output={output}\n\n")


if __name__ == "__main__":
    # test env
    MY_RANK = os.environ.get("RANK", None)
    MY_WORLD_SIZE = os.environ.get("WORLD_SIZE", None)
    MASTER_ADDR = os.environ.get("MASTER_ADDR", None)
    MASTER_PORT = os.environ.get("MASTER_PORT", None)
    if any([x is None for x in [MY_RANK, MY_WORLD_SIZE, MASTER_ADDR, MASTER_PORT]]):
        print("RANK or WORLD_SIZE or MASTER_ADDR or MASTER_PORT is not set.")
        sys.exit(1)

    if int(MY_RANK) == 0 or int(MY_RANK) == int(MY_WORLD_SIZE) - 1:
        print(f"Initializing process group from {socket.gethostname()} at {datetime.now()}.")
    init_start = time.time()
    # initialize the distributed environment
    # create a process group and set the communication backend to be NCCL
    # assign each process a unique rank and initialize the network connections
    store = None
    verbose = 0
    verbose = 1 if (int(MY_RANK) == 0 or int(MY_RANK) == int(MY_WORLD_SIZE) - 1) else 0
    store = RedisStore(int(MY_RANK) == 0, num_shards=1, verbose=verbose)
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=30),
        world_size=int(MY_WORLD_SIZE),
        rank=int(MY_RANK),
        store=store,
    )
    init_done = time.time()
    if int(MY_RANK) == 0 or int(MY_RANK) == int(MY_WORLD_SIZE) - 1:
        print(f"\ninit_process_group done; elapsed={init_done - init_start}", flush=True)
        if store:
            store.dump_and_reset_stats()
    time.sleep(1)

    if int(MY_RANK) == 0 or int(MY_RANK) == int(MY_WORLD_SIZE) - 1:
        print(f"\nEntering barrier1... at {datetime.now()}", flush=True)
    barrier1_start = time.time()
    torch.distributed.barrier()
    barrier1_end = time.time()
    if int(MY_RANK) == 0 or int(MY_RANK) == int(MY_WORLD_SIZE) - 1:
        print(f"\nPassing barrier1; elapsed={barrier1_end - barrier1_start}", flush=True)
        if store:
            store.dump_and_reset_stats()
    time.sleep(1)

    if int(MY_RANK) == 0 or int(MY_RANK) == int(MY_WORLD_SIZE) - 1:
        print(f"\nCreating nccl group... at {datetime.now()}", flush=True)
    group_start = time.time()
    group = torch.distributed.new_group(range(int(MY_WORLD_SIZE)), backend="nccl")
    group_end = time.time()
    if int(MY_RANK) == 0 or int(MY_RANK) == int(MY_WORLD_SIZE) - 1:
        print(f"\nCreating nccl group done; elapsed={group_end - group_start}", flush=True)
        if store:
            store.dump_and_reset_stats()
    time.sleep(1)

    if int(MY_RANK) == 0 or int(MY_RANK) == int(MY_WORLD_SIZE) - 1:
        print(f"\nCreating gloo group... at {datetime.now()}", flush=True)
    gloo_group_start = time.time()
    gloo_group = torch.distributed.new_group(range(int(MY_WORLD_SIZE)), backend="gloo")
    gloo_group_end = time.time()
    if int(MY_RANK) == 0 or int(MY_RANK) == int(MY_WORLD_SIZE) - 1:
        print(
            f"\nCreating gloo group done; elapsed={gloo_group_end - gloo_group_start}",
            flush=True,
        )
        if store:
            store.dump_and_reset_stats()
    time.sleep(1)

    # Run a single training step
    # demo_data_parallel()

    if int(MY_RANK) == 0 or int(MY_RANK) == int(MY_WORLD_SIZE) - 1:
        print(f"\nEntering barrier2... at {datetime.now()}", flush=True)
    barrier2_start = time.time()
    torch.distributed.barrier()
    barrier2_end = time.time()
    if int(MY_RANK) == 0 or int(MY_RANK) == int(MY_WORLD_SIZE) - 1:
        print(f"\nPassing barrier2; elapsed={barrier2_end - barrier2_start}", flush=True)
        if store:
            store.dump_stats()

    # Run a multi-gpu all reduce
    torch.distributed.barrier()
    demo_all_reduce()
    time.sleep(1)
    demo_all_reduce()

    torch.distributed.destroy_process_group()
