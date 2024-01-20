#include <cmath>
#include <cstdint>
#include <iostream>
#include <cuda.h>

// Kernel function to add the elements of two arrays
__global__ void add(size_t n, float *x, float *y)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride)
    {
        y[i] = x[i] + y[i];
    }
}

int main(void)
{
    size_t N = (size_t)(2) << 30; // 2G x 4B x 2
    float *x, *y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (size_t i = 0; i < N; ++i)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaEvent_t ckpt1, ckpt2, ckpt3;
    cudaEventCreate(&ckpt1);
    cudaEventCreate(&ckpt2);
    cudaEventCreate(&ckpt3);

    std::cout << "Initialized " << N << " elements, GPU kernel start" << std::endl;

    size_t blockSize = 1024;
    size_t numBlocks = (N + blockSize - 1) / blockSize;

    cudaEventRecord(ckpt1);

    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaEventRecord(ckpt2);

    for (int k = 0; k < 100; ++k) {
        add<<<numBlocks, blockSize>>>(N, x, y);
    }
    cudaEventRecord(ckpt3);

    cudaDeviceSynchronize();
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, ckpt1, ckpt2);
    std::cout << "Elapsed time " << elapsed / 1.0e3 << " seconds - initial" << std::endl;
    cudaEventElapsedTime(&elapsed, ckpt2, ckpt3);
    std::cout << "Elapsed time " << elapsed / 1.0e5 << " seconds - warmedup" << std::endl;

    float maxError = 0.0f;
    for (size_t i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    cudaEventDestroy(ckpt1);
    cudaEventDestroy(ckpt2);
    cudaEventDestroy(ckpt3);

    cudaFree(x);
    cudaFree(y);

    return int(maxError);
}