#!/usr/bin/env python3

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch location: {torch.__file__}")

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    print("CUDA Version:", torch.version.cuda)
    print("GPU Count:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
    if torch.cuda.nccl.is_available(torch.randn(1).cuda()):
        print("NCCL is available.")
        print("NCCL Version:", torch.cuda.nccl.version())
    else:
        print("NCCL is not available.")
else:
    print("CUDA is not available. Using CPU.")
