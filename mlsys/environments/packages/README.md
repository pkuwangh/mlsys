# Build Packages

## NCCL

```bash
cd nccl
# on GraceHopper, need latest version of cuda-toolkit
make -j src.build CUDA_HOME=<path-to-cuda> NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```
