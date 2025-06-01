# Build Packages

## NCCL

```bash
cd nccl
# on GraceHopper, need latest version of cuda-toolkit
make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

## Pytorch

Install Dependencies

```bash
# system dependencies
sudo apt-get install cmake ninja-build
sudo apt-get install libcudnn9-cuda-12 libcudnn9-dev-cuda-12
# python dependencies
cd pytorch
# newer GPU may require new version of CUDA and in turn unreleased version of pytorch.
git submodule update --init --recursive
pip3 install -r requirements.txt
```

Build pytorch

```bash
# compile pytorch with new C++ ABI enabled
export _GLIBCXX_USE_CXX11_ABI=1
# build!
USE_CUDA=1 TORCH_CUDA_ARCH_LIST="9.0" USE_NCCL=1 USE_SYSTEM_NCCL=1 USE_UCC=0 python3 setup.py develop
# USE_CUDA=1 TORCH_CUDA_ARCH_LIST="9.0" USE_NCCL=1 USE_SYSTEM_NCCL=1 USE_UCC=0 python3 setup.py install
```
