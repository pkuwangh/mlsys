#!/bin/bash

pip3 uninstall torch torchvision torchaudio

# with NCCL & UCC enabled
pip3 install http://10.31.241.55/nvdl/datasets/pip-scratch/nvidia-pytorch/torch-2.3.0a0+ebedce2.nv99.99.sbsa.12772057-cp310-cp310-linux_aarch64.whl
