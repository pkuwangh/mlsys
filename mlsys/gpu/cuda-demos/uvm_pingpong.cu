#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <thread>

// CUDA error handling
#define CU_ASSERT_RESULT(x)                                                                                            \
    do {                                                                                                               \
        CUresult cuResult = (x);                                                                                       \
        if ((cuResult) != CUDA_SUCCESS) {                                                                              \
            const char *errDescStr, *errNameStr;                                                                       \
            cuGetErrorString(cuResult, &errDescStr);                                                                   \
            cuGetErrorName(cuResult, &errNameStr);                                                                     \
            fprintf(stderr, "[%s] %s in expr %s in %s() : %s:%d\n", errNameStr, errDescStr, #x, __PRETTY_FUNCTION__,   \
                    __FILE__, __LINE__);                                                                               \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    } while (0)

#define CU_ASSERT_ERROR(x)                                                                                             \
    do {                                                                                                               \
        cudaError_t cuError = (x);                                                                                     \
        if ((cuError) != cudaSuccess) {                                                                                \
            fprintf(stderr, "[%s] %s in expr %s in %s() : %s:%D\n", cudaGetErrorName(cuError),                         \
                    cudaGetErrorString(cuError), #x, __PRETTY_FUNCTION__, __FILE__, __LINE__);                         \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    } while (0)


// CPU sync
void sync_to_gpu(void* buffer, size_t data_offset, size_t data_size, void* data_cpu, size_t seqnum_offset,
                 int64_t seqnum) {
    // write data to buffer
    std::memcpy((char *)buffer + data_offset, data_cpu, data_size);
    // ensure all writes to buffer are visible
    __sync_synchronize();
    // write seqnum to buffer
    *(int64_t *)((char *)buffer + seqnum_offset) = seqnum;
    // push it out eagerly?
    // __sync_synchronize();
}

void sync_from_gpu(void* buffer, size_t seqnum_offset, int64_t seqnum) {
    // wait until GPU writes the seqnum
    volatile int64_t* buffer_seqnum = (int64_t *)((char *)buffer + seqnum_offset);
    while (*buffer_seqnum != seqnum) {
        // spin
    }
}

// GPU sync
__global__ void sync_to_cpu(void* buffer, size_t data_offset, size_t data_size, void* data_gpu, size_t seqnum_offset,
                            int64_t seqnum) {
    // write data to buffer; assume one block is enough
    size_t tid = threadIdx.x;
    volatile float* buffer_data = (float *)buffer + data_offset / sizeof(float);
    for (size_t i = tid; i < data_size / sizeof(float); i += blockDim.x) {
        *(buffer_data + i) = ((float *)data_gpu)[i];
        ((float*)data_gpu)[i] = 0.0f;   // clear for next iteration
    }
    // ensure all writes to buffer are visible to CPU
    __threadfence_system();
    // write seqnum to buffer
    volatile int64_t* buffer_seqnum = (int64_t *)buffer + seqnum_offset / sizeof(int64_t);
    if (tid == 0) {
        *buffer_seqnum = seqnum;
    }
    // push it out eagerly?
    // __threadfence_system();
}

__global__ void sync_from_cpu(void* buffer, size_t seqnum_offset, int64_t seqnum, int64_t timeout) {
    // wait until CPU writes the seqnum
    int64_t start = clock64();
    volatile int64_t* buffer_seqnum = (int64_t *)buffer + seqnum_offset / sizeof(int64_t);
    while (*buffer_seqnum != seqnum) {
        // spin
        if (clock64() - start > timeout) { // timeout after ~1 second
            printf("Timeout waiting for CPU to write seqnum %lli\n", seqnum);
            asm("trap;");
        }
    }
}

__global__ void sync_on_gpu(void *buffer, size_t data_offset, size_t data_size, void *data_gpu, size_t seqnum_offset,
                            int64_t seqnum, int64_t timeout) {
    // write data to buffer; assume one block is enough
    size_t tid = threadIdx.x;
    volatile float* buffer_data = (float *)buffer + data_offset / sizeof(float);
    for (size_t i = tid; i < data_size / sizeof(float); i += blockDim.x) {
        *(buffer_data + i) = ((float *)data_gpu)[i];
        ((float*)data_gpu)[i] = 0.0f;   // clear for next iteration
    }
    // ensure all writes to buffer are visible to CPU
    __threadfence_system();
    // write seqnum to buffer
    volatile int64_t* buffer_seqnum = (int64_t *)buffer + seqnum_offset / sizeof(int64_t);
    if (tid == 0) {
        *buffer_seqnum = seqnum;
    }
    // push it out eagerly?
    // __threadfence_system();

    // wait until CPU writes the seqnum
    if (tid == 0) {
        int64_t start = clock64();
        while (*buffer_seqnum != (seqnum + 1)) {
            // spin
            if (clock64() - start > timeout) { // timeout after ~1 second
                printf("Timeout waiting for CPU to write seqnum %lli\n", (seqnum + 1));
                asm("trap;");
            }
        }
    }
}

// accessible from both cpu & gpu, non-migratable
class SyncBuffer {
  public:
    size_t size = 0;
    size_t alloc_size = 0;
    size_t page_size = 0;
    CUmemGenericAllocationHandle handle;
    void* buffer = nullptr;

    SyncBuffer(size_t x_size) : size(x_size) {
        int cuda_dev;
        CUdevice current_dev;
        CU_ASSERT_ERROR(cudaGetDevice(&cuda_dev));
        CU_ASSERT_RESULT(cuDeviceGet(&current_dev, cuda_dev));
        // get cpu NUMA id and set location type
        int cpu_numa_node_id = -1;
        CU_ASSERT_RESULT(cuDeviceGetAttribute(&cpu_numa_node_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, current_dev));
        fprintf(stdout, "CUDA device=%d/%d hostNumaId=%d\n", cuda_dev, current_dev, cpu_numa_node_id);
        // memory allocation property
        CUmemAllocationProp memprop = {};
        memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        memprop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        memprop.location.id = cpu_numa_node_id;
         // size & granularity
        CU_ASSERT_RESULT(cuMemGetAllocationGranularity(&page_size, &memprop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        size_t alloc_size = size;
        if (size % page_size > 0) {
            alloc_size = page_size * (size / page_size + 1);
        }
        fprintf(stdout, "Requested size=%zu granularity=%zu adjusted size=%zu\n", size, page_size, alloc_size);
        // physical memory allocation
        CU_ASSERT_RESULT(cuMemCreate(&handle, alloc_size, &memprop, 0));
        // reserve an address space and map it to a pointer
        CU_ASSERT_RESULT(cuMemAddressReserve((CUdeviceptr *)(&buffer), alloc_size, 0, 0, 0));
        CU_ASSERT_RESULT(cuMemMap((CUdeviceptr)buffer, alloc_size, 0, handle, 0));
        // explicitly protect mapped VA ranges
        CUmemAccessDesc access_desc[2] = {{}};
        access_desc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc[0].location.id = current_dev;
        access_desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        access_desc[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        access_desc[1].location.id = cpu_numa_node_id;
        access_desc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_ASSERT_RESULT(cuMemSetAccess((CUdeviceptr)buffer, alloc_size, access_desc, 2));
        // clear it
        CU_ASSERT_ERROR(cudaMemset(buffer, 0, alloc_size));
        CU_ASSERT_ERROR(cudaDeviceSynchronize());
    }
};



// reduce kernel
__global__ void reduce(const float* __restrict__ g_in, size_t n, float* g_out) {
    extern __shared__ float sdata[]; // shared memory

    size_t tid = threadIdx.x;
    size_t idx = (size_t)(blockIdx.x) * (size_t)(blockDim.x) * 2 + tid;

    // load two elements per thread (to reduce divergence)
    float sum = 0.0f;
    if (idx < n)
        sum += g_in[idx];
    if (idx + blockDim.x < n)
        sum += g_in[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // do reduction in shared memory
    for (size_t s = (size_t)(blockDim.x) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // add to global sum
    if (tid == 0) {
        atomicAdd(g_out, sdata[0]);
    }
}

// vector-add kernel
__global__ void saxpy(const float* __restrict x, float* y, float* a, size_t n) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        y[i] = (*a) * x[i] + y[i];
    }
}

// init kernel
__global__ void init(float* x, float* y, float* sum, size_t n) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    if (index == 0) {
        *sum = 0.0f;
    }
}


void run_default(float* x, float* y, float* sum, int num_elements, int block_size, int grid_size, size_t shm_bytes, int total_iters) {
    float sum_cpu = 0;

    auto start_timestamp = std::chrono::high_resolution_clock::now();

    int num_iters = 0;
    while (num_iters < total_iters) {
        // reduce kernel
        CU_ASSERT_ERROR(cudaMemset(sum, 0, sizeof(float)));
        reduce<<<grid_size, block_size, shm_bytes>>>(x, num_elements, sum);

        // copy back to cpu
        cudaMemcpyAsync(&sum_cpu, sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(0);

        // serial work on CPU
        float a_cpu = sum_cpu / (float)num_elements / 2 + (float)num_iters;
        // fprintf(stdout, "iter=%d reduce sum=%f a=%f\n", num_iters, sum_cpu, a_cpu);
        // std::this_thread::sleep_for(std::chrono::microseconds(1000)); // simulate cpu work

        // copy a to gpu
        cudaMemcpyAsync(sum, &a_cpu, sizeof(float), cudaMemcpyHostToDevice);

        // saxpy kernel
        saxpy<<<grid_size, block_size>>>(x, y, sum, num_elements);

        num_iters += 1;
    }

    CU_ASSERT_ERROR(cudaDeviceSynchronize());
    auto end_timestamp = std::chrono::high_resolution_clock::now();
    double total_seconds = std::chrono::duration<double>(end_timestamp - start_timestamp).count();

    fprintf(stdout, "%d iters elapsed in %.3f ms (%.3f us / iter), y[0]=%.1f y[1]=%.1f, ... y[n-1]=%.1f\n",
            total_iters, total_seconds * 1e3, total_seconds * 1e6 / total_iters, y[0], y[1], y[num_elements - 1]);
}


void run_uvm(float* x, float* y, float* sum, int num_elements, int block_size, int grid_size, size_t shm_bytes, int total_iters, SyncBuffer& sync_buffer, size_t seqnum_offset, size_t data_offset, float* data_ptr) {
    const int64_t seqnum_start = 1;
    const int num_inflight_iters = 10;
    const int max_submits_per_loop = 2;

    auto start_timestamp = std::chrono::high_resolution_clock::now();

    int seqnum_cpu = seqnum_start;
    int seqnum_gpu = seqnum_start;
    int num_iters_cpu = 0;
    int num_iters_gpu = 0;
    while (num_iters_cpu < total_iters) {
        int num_submitted = 0;
        while (num_iters_gpu < total_iters && num_iters_gpu - num_iters_cpu < num_inflight_iters &&
               num_submitted < max_submits_per_loop) {
            // fprintf(stdout, "Submitting iter %d (seqnum=%lli) to GPU...\n", num_iters_gpu, seqnum_gpu);
            // reduce kernel on GPU
            reduce<<<grid_size, block_size, shm_bytes>>>(x, num_elements, sum);

            // sync on GPU
            sync_on_gpu<<<1, block_size>>>(sync_buffer.buffer, data_offset, sizeof(float), sum, seqnum_offset, seqnum_gpu, 10e9);
            seqnum_gpu += 2;

            // // sync to CPU
            // sync_to_cpu<<<1, block_size>>>(sync_buffer.buffer, data_offset, sizeof(float), sum, seqnum_offset, seqnum_gpu);
            // seqnum_gpu += 1;

            // // sync from cpu
            // sync_from_cpu<<<1, 1>>>(sync_buffer.buffer, seqnum_offset, seqnum_gpu, 10e9);
            // seqnum_gpu += 1;

            // saxpy kernel on GPU
            saxpy<<<grid_size, block_size>>>(x, y, data_ptr, num_elements);

            num_iters_gpu += 1;
            num_submitted += 1;
        }

        // cpu-side work
        // wait for reduce to finish and read result
        // fprintf(stdout, "Waiting for seqnum %d from GPU...\n", seqnum_cpu);
        sync_from_gpu(sync_buffer.buffer, seqnum_offset, seqnum_cpu);
        seqnum_cpu += 1;

        // serial work on CPU
        float a_cpu = *data_ptr / (float)num_elements / 2 + (float)num_iters_cpu;
        // fprintf(stdout, "iter=%d reduce sum=%f a=%f\n", num_iters_cpu, *data_ptr, a_cpu);
        // std::this_thread::sleep_for(std::chrono::microseconds(1000)); // simulate cpu work

        // fprintf(stdout, "About to send seqnum %d to GPU...\n", seqnum_cpu);
        sync_to_gpu(sync_buffer.buffer, data_offset, sizeof(float), &a_cpu, seqnum_offset, seqnum_cpu);
        seqnum_cpu += 1;

        num_iters_cpu += 1;
    }

    CU_ASSERT_ERROR(cudaDeviceSynchronize());
    auto end_timestamp = std::chrono::high_resolution_clock::now();
    double total_seconds = std::chrono::duration<double>(end_timestamp - start_timestamp).count();

    fprintf(stdout, "%d iters elapsed in %.3f ms (%.3f us / iter), y[0]=%.1f y[1]=%.1f, ... y[n-1]=%.1f\n",
            total_iters, total_seconds * 1e3, total_seconds * 1e6 / total_iters, y[0], y[1], y[num_elements - 1]);
}

void init_values(float* x, float* y, float* sum, size_t n) {
    // initialize
    init<<<256, 64>>>(x, y, sum, n);
    CU_ASSERT_ERROR(cudaGetLastError());
    CU_ASSERT_ERROR(cudaDeviceSynchronize());
    fprintf(stdout, "After init, x[0]=%f x[1]=%f, ... x[n-1]=%f\n", x[0], x[1], x[n - 1]);
}

int main() {
    // make a sync buffer
    SyncBuffer sync_buffer((size_t)(4) << 20);  // 4MB
    const size_t seqnum_offset = 0;
    const size_t data_offset = sync_buffer.page_size;
    int64_t* seqnum_ptr = (int64_t *)((char *)sync_buffer.buffer + seqnum_offset);
    float* data_ptr = (float *)((char *)sync_buffer.buffer + data_offset);
    fprintf(stdout, "starting seqnum=%lli data[0]=%f\n", *seqnum_ptr, *data_ptr);

    float* x = nullptr;
    float* y = nullptr;
    float* sum = nullptr;
    // const size_t num_elements = 1 << 28; // 256M elements ~1GB
    const size_t num_elements = 1 << 18; // 256K elements ~1MB
    CU_ASSERT_ERROR(cudaMallocManaged(&x, num_elements * sizeof(float)));
    CU_ASSERT_ERROR(cudaMallocManaged(&y, num_elements * sizeof(float)));
    CU_ASSERT_ERROR(cudaMallocManaged(&sum, sizeof(float)));

    const int block_size = 256;
    const int grid_size = (num_elements + block_size * 2 - 1) / (block_size * 2);
    const size_t shm_bytes = block_size * sizeof(float);
    const int total_iters = 2000;

    init_values(x, y, sum, num_elements);
    run_default(x, y, sum, num_elements, block_size, grid_size, shm_bytes, total_iters);

    init_values(x, y, sum, num_elements);
    run_uvm(x, y, sum, num_elements, block_size, grid_size, shm_bytes, total_iters, sync_buffer, seqnum_offset, data_offset, data_ptr);

    return 0;
}
