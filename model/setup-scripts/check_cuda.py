#!/usr/bin/env python3

import torch

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    print("GPU Count:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")