#!/usr/bin/env python3

import os
import sys
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel


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
    MY_RANK = os.environ.get("RANK", "0")
    MY_WORLD_SIZE = os.environ.get("WORLD_SIZE", "1")
    if not MY_RANK or not MY_WORLD_SIZE:
        print("RANK or WORLD_SIZE is not set.")
        sys.exit(1)

    print("Initializing process group.")
    # initialize the distributed environment
    # create a process group and set the communication backend to be NCCL
    # assign each process a unique rank and initialize the network connections
    torch.distributed.init_process_group("nccl")

    # Run a single training step
    demo_data_parallel()
    torch.distributed.barrier()

    # Run a multi-gpu all reduce
    demo_all_reduce()
    torch.distributed.barrier()

    torch.distributed.destroy_process_group()
