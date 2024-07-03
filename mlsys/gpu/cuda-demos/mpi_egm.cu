#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <mpi.h>
#include <nvml.h>
#include <stdexcept>
#include <unistd.h>

// CUDA error handling
#define CU_ASSERT_RESULT(x)                                                    \
    do {                                                                       \
        CUresult cuResult = (x);                                               \
        if ((cuResult) != CUDA_SUCCESS) {                                      \
            const char *errDescStr, *errNameStr;                               \
            cuGetErrorString(cuResult, &errDescStr);                           \
            cuGetErrorName(cuResult, &errNameStr);                             \
            fprintf(stderr, "[%s] %s in expression %s in %s() : %s:%d\n",      \
                    errNameStr, errDescStr, #x, __PRETTY_FUNCTION__, __FILE__, \
                    __LINE__);                                                 \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CU_ASSERT_ERROR(x)                                                     \
    do {                                                                       \
        cudaError_t cuError = (x);                                             \
        if ((cuError) != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA runtime error [%s] %s\n",                    \
                    cudaGetErrorName(cuError), cudaGetErrorString(cuError));   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define NVML_ASSERT(x)                                                         \
    do {                                                                       \
        nvmlReturn_t nvmlResult = (x);                                         \
        if ((nvmlResult) != NVML_SUCCESS) {                                    \
            fprintf(stderr, "NVML error %d:%s\n", nvmlResult,                  \
                    nvmlErrorString(nvmlResult));                              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// GPU kernels
__global__ void init(float *buffer, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        buffer[i] = static_cast<float>(i);
    }
}

__global__ void reduce(float *input, float *output, int size) {
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory, handling out-of-bounds elements
    sdata[tid] = (idx < size) ? input[idx] : 0;
    __syncthreads();

    // Perform tree-based reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main(int argc, char **argv) {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    printf("Rank %d on %s out of %d total ranks\n", world_rank, hostname,
           world_size);

    // cuda initialzation
    CU_ASSERT_RESULT(cuInit(0));
    NVML_ASSERT(nvmlInit());
    MPI_Barrier(MPI_COMM_WORLD);

    // get device handle
    int cudaDev;
    CUdevice currentDev;
    CU_ASSERT_ERROR(cudaGetDevice(&cudaDev));
    CU_ASSERT_RESULT(cuDeviceGet(&currentDev, cudaDev));
    printf("get cuda device %d/%d\n", cudaDev, currentDev);

    // get cpu NUMA id and set location type
    int cpuNumaNodeId = -1;
    CUmemLocationType location_type = CU_MEM_LOCATION_TYPE_DEVICE;
    CU_ASSERT_RESULT(cuDeviceGetAttribute(
        &cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));
    bool hostMem = (cpuNumaNodeId != -1);
    location_type = hostMem ? CU_MEM_LOCATION_TYPE_HOST_NUMA : location_type;
    printf("hostMem-%d, host numa ID=%d\n", hostMem, cpuNumaNodeId);

    // memory allocation property
    CUmemAllocationProp memprop = {};
    memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memprop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    memprop.location.type = location_type;
    memprop.location.id = hostMem ? cpuNumaNodeId : currentDev;

    // size & granularity
    size_t allocSize = ((size_t)(128) << 20) * sizeof(float);
    size_t granu = 0;
    CU_ASSERT_RESULT(cuMemGetAllocationGranularity(
        &granu, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    size_t origSize = allocSize;
    if (allocSize % granu > 0) {
        allocSize = granu * (allocSize / granu + 1);
    }
    printf("requested size=%zu, padded alloc size=%zu, granularity=%zu\n",
           origSize, allocSize, granu);

    // allocate physical memory
    CUmemGenericAllocationHandle allocHandle;
    CUmemFabricHandle fabricHandle;
    if (world_rank == 0) {
        CU_ASSERT_RESULT(cuMemCreate(&allocHandle, allocSize, &memprop, 0));
        CU_ASSERT_RESULT(cuMemExportToShareableHandle(
            &fabricHandle, allocHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
        printf("allocHandle=%llx\n", allocHandle);
        printf("fabricHandle=");
        for (auto x : fabricHandle.data) {
            printf("%02x", x);
        }
        printf("\n");
    }

    // broadcast fabric handle
    MPI_Bcast(&fabricHandle, sizeof(fabricHandle), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (world_rank != 0) {
        printf("fabricHandle=");
        for (auto x : fabricHandle.data) {
            printf("%02x", x);
        }
        printf("\n");
        CU_ASSERT_RESULT(cuMemImportFromShareableHandle(
            &allocHandle, (void *)&fabricHandle, CU_MEM_HANDLE_TYPE_FABRIC));
        printf("allocHandle=%llx\n", allocHandle);
    }
    // just for safety: wait after every rank is done importing
    MPI_Barrier(MPI_COMM_WORLD);

    // map the memory
    float *buffer_egm;
    CU_ASSERT_RESULT(
        cuMemAddressReserve((CUdeviceptr *)&buffer_egm, allocSize, 0, 0, 0));
    CU_ASSERT_RESULT(
        cuMemMap((CUdeviceptr)buffer_egm, allocSize, 0, allocHandle, 0));

    // explicitly protect mapped VA ranges
    CUmemAccessDesc accessDesc;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    // always make it accessible to the device
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = currentDev;
    CU_ASSERT_RESULT(
        cuMemSetAccess((CUdeviceptr)buffer_egm, allocSize, &accessDesc, 1));
    if (world_rank == 0 && location_type != CU_MEM_LOCATION_TYPE_DEVICE) {
        accessDesc.location.type = location_type;
        accessDesc.location.id = hostMem ? cpuNumaNodeId : currentDev;
        CU_ASSERT_RESULT(
            cuMemSetAccess((CUdeviceptr)buffer_egm, allocSize, &accessDesc, 1));
    }

    // Make sure every rank is done with mapping the fabric allocation
    MPI_Barrier(MPI_COMM_WORLD);

    // write data from one rank and read from other ranks
    int writer_rank = 0;
    int reader_rank = 1;

    size_t num_elements = allocSize / sizeof(float);
    int block_size = 256;
    int num_blocks = (num_elements + (size_t)(block_size - 1)) / block_size;
    printf("num_elements=%zu, block_size=%d, num_blocks=%d\n", num_elements,
           block_size, num_blocks);
    // write from writer_rank
    if (world_rank == writer_rank) {
        printf("Write data from rank=%d to EGM buffer\n", world_rank);
        init<<<num_blocks, block_size>>>(buffer_egm, num_elements);
        CU_ASSERT_ERROR(cudaDeviceSynchronize());
        CU_ASSERT_ERROR(cudaGetLastError());
    }

    // sync
    printf("Rank %d waiting at barrier\n", world_rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    // extra safety
    CU_ASSERT_ERROR(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // read from other ranks
    if (world_rank == reader_rank) {
        float *sum;
        sum = (float *)malloc(sizeof(float) * num_blocks);

        // verify
        printf("Reading from rank=%d to sum up first 10 elements\n",
               world_rank);
        reduce<<<num_blocks, block_size, block_size * sizeof(float)>>>(
            buffer_egm, sum, 10);
        CU_ASSERT_ERROR(cudaDeviceSynchronize());
        CU_ASSERT_ERROR(cudaGetLastError());
        printf("sum[0]=%f\n", sum[0]);

        float *buffer_local;
        CU_ASSERT_ERROR(cudaMalloc(&buffer_local, allocSize));
        cudaEvent_t ckpt1, ckpt2, ckpt3;
        cudaEventCreate(&ckpt1);
        cudaEventCreate(&ckpt2);
        cudaEventCreate(&ckpt3);
        // copy
        printf(
            "cudaMemcpy buffer of size=%zuMB from rank=%d EGM to rank=%d HBM\n",
            allocSize / 1024 / 1024, 0, world_rank);
        cudaEventRecord(ckpt1);
        CU_ASSERT_ERROR(cudaMemcpy(buffer_local, buffer_egm, allocSize,
                                   cudaMemcpyDeviceToDevice));
        cudaEventRecord(ckpt2);
        CU_ASSERT_ERROR(cudaDeviceSynchronize());
        cudaEventRecord(ckpt3);
        float elapsed2 = 0, elapsed3 = 0;
        cudaEventElapsedTime(&elapsed2, ckpt1, ckpt2);
        cudaEventElapsedTime(&elapsed3, ckpt1, ckpt3);
        printf("Elapsed time %f/%f mili-seconds\n", elapsed2, elapsed3);
        printf("Cross-node EGM->HBM copy bandwidth: %f GB/s\n",
               (allocSize / 1024.0 / 1024 / 1024) / (elapsed3 / 1000.0));

        // verify after copy
        printf("Verifying data after cudaMemcpy from rank=%d\n", world_rank);
        reduce<<<num_blocks, block_size, block_size * sizeof(float)>>>(
            buffer_local, sum, 10);
        CU_ASSERT_ERROR(cudaDeviceSynchronize());
        CU_ASSERT_ERROR(cudaGetLastError());
        printf("sum[0]=%f\n", sum[0]);
    }

    // cleanup
    MPI_Barrier(MPI_COMM_WORLD);

    printf("Cleaning up\n");
    CU_ASSERT_RESULT(cuMemUnmap((CUdeviceptr)buffer_egm, allocSize));
    CU_ASSERT_RESULT(cuMemRelease(allocHandle));
    CU_ASSERT_RESULT(cuMemAddressFree((CUdeviceptr)buffer_egm, allocSize));

    MPI_Finalize();

    printf("Done!\n");
    return 0;
}
