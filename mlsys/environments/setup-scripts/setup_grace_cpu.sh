#!/bin/bash

sudo apt-get install libcudnn8
sudo apt-get install libcudnn8-dev
sudo apt-get install libopenblas-dev
sudo apt-get install mpich

pip3 uninstall torch torchvision torchaudio
pip3 install numpy packaging
pip3 install http://10.31.241.55/nvdl/datasets/pip-scratch/jp/v60dp/pytorch/torch-2.3.0a0+ebedce2.nv24.02-cp310-cp310-linux_aarch64.whl
pip3 install pynvml pycuda
