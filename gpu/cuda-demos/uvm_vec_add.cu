#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <boost/program_options.hpp>
#include <cuda.h>

namespace po = boost::program_options;

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

int benchmark(const std::string &allocator)
{
    size_t N = (size_t)(2) << 30; // 2G x 4B x 2
    float *x, *y;

    if (allocator == "cudaMallocManaged")
    {
        cudaMallocManaged(&x, N * sizeof(float));
        cudaMallocManaged(&y, N * sizeof(float));
    }
    else if (allocator == "cudaMallocHost")
    {
        cudaMallocHost(&x, N * sizeof(float));
        cudaMallocHost(&y, N * sizeof(float));
    }
    else if (allocator == "malloc")
    {
        x = (float *)(malloc(N * sizeof(float)));
        y = (float *)(malloc(N * sizeof(float)));
    }
    else
    {
        std::cout << "Unknown allocator " << allocator << std::endl;
        return 1;
    }

    for (size_t i = 0; i < N; ++i)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaEvent_t ckpt1, ckpt2, ckpt3;
    cudaEventCreate(&ckpt1);
    cudaEventCreate(&ckpt2);
    cudaEventCreate(&ckpt3);

    std::cout << "Initialized " << N << " elements using " << allocator << std::endl;

    size_t blockSize = 1024;
    size_t numBlocks = (N + blockSize - 1) / blockSize;

    cudaEventRecord(ckpt1);

    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaEventRecord(ckpt2);

    for (int k = 0; k < 100; ++k)
    {
        add<<<numBlocks, blockSize>>>(N, x, y);
    }
    cudaEventRecord(ckpt3);

    cudaDeviceSynchronize();
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, ckpt1, ckpt2);
    std::cout << "Elapsed time " << int(elapsed) << " mili-seconds - initial" << std::endl;
    cudaEventElapsedTime(&elapsed, ckpt2, ckpt3);
    std::cout << "Elapsed time " << int(elapsed / 100) << " mili-seconds - warmedup" << std::endl;

    float maxError = 0.0f;
    for (size_t i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    cudaEventDestroy(ckpt1);
    cudaEventDestroy(ckpt2);
    cudaEventDestroy(ckpt3);

    if (allocator == "malloc")
    {
        free(x);
        free(y);
    }
    else
    {
        cudaFree(x);
        cudaFree(y);
    }
    return int(maxError);
}

int main(int argc, char **argv)
{
    std::set<std::string> avail_allocators({"cudaMallocHost",
                                            "cuMemCreate",
                                            "cudaMallocManaged",
                                            "malloc"});
    std::string allocator;
    po::options_description all_opts("uvm_vec_add CLI");
    all_opts.add_options()("help,h", "Help message");
    all_opts.add_options()("list,l", "list available allocator options");
    all_opts.add_options()("allocator,a", po::value<std::string>(&allocator)->default_value("all"), "Which memory allocation API to use");

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, all_opts), vm);
        po::notify(vm);
    }
    catch (...)
    {
        std::cout << "Error: Invalid Arguments " << std::endl;
        std::cout << all_opts << std::endl;
        return 1;
    }
    if (vm.count("help"))
    {
        std::cout << all_opts << std::endl;
        return 0;
    }
    if (vm.count("list"))
    {
        for (const std::string &x : avail_allocators)
        {
            std::cout << x << std::endl;
        }
        return 0;
    }

    int totalError = 0;
    if (allocator == "all")
    {
        for (const std::string &x : avail_allocators)
        {
            totalError += benchmark(x);
        }
    }
    else
    {
        totalError += benchmark(allocator);
    }

    return int(totalError);
}