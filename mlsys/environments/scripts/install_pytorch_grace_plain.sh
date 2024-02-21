#!/bin/bash

pip3 uninstall torch torchvision torchaudio

# without NCCL
pip3 install http://10.31.241.55/nvdl/datasets/pip-scratch/jp/v60dp/pytorch/torch-2.3.0a0+ebedce2.nv24.02-cp310-cp310-linux_aarch64.whl
