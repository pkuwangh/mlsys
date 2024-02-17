#!/bin/bash

sudo apt-get install libcudnn9-cuda-12 libcudnn9-dev-cuda-12
sudo apt-get install libopenblas-dev mpich

pip3 uninstall torch torchvision torchaudio
pip3 install numpy packaging

# without nccl support
# pip3 install http://10.31.241.55/nvdl/datasets/pip-scratch/jp/v60dp/pytorch/torch-2.2.0a0+81ea7a4.nv24.01-cp310-cp310-linux_aarch64.whl

# with nccl support
pip3 install http://10.31.241.55/nvdl/datasets/pip-scratch/nvidia-pytorch/torch-2.3.0a0+ebedce2.nv99.99.sbsa.12772057-cp310-cp310-linux_aarch64.whl

