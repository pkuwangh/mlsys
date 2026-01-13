#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <string>

#define WARPSIZE 32

using bf16 = __nv_bfloat16;

inline void checkCuda(cudaError_t err, const char *message) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

inline void fillSequential(float *data, int count) {
    for (int i = 0; i < count; ++i) {
        data[i] = static_cast<float>(i);
    }
}

inline void fillSequential(bf16 *data, int count) {
    for (int i = 0; i < count; ++i) {
        data[i] = __float2bfloat16(static_cast<float>(i));
    }
}

inline void fillSequentialColMajor(bf16 *data, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[j * rows + i] = __float2bfloat16(static_cast<float>(i * cols + j));
        }
    }
}

inline dim3 makeGrid2D(int rows, int cols, dim3 block) {
    // x, y (, z)
    return dim3((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
}

inline void printMatrix(const float *matrix, int rows, int cols, std::string matrix_name) {
    std::printf("%s:\n", matrix_name.c_str());
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::printf("%8.1f", matrix[i * cols + j]);
        }
        std::printf("\n");
    }
}

class MatmulBuffers {
  public:
    int M;
    int K;
    int N;
    std::vector<float> hA;
    std::vector<float> hB;
    std::vector<float> hC;
    float *dA{};
    float *dB{};
    float *dC{};
    std::vector<bf16> hA_bf16;
    std::vector<bf16> hB_bf16;
    std::vector<bf16> hC_bf16;
    bf16 *dA_bf16{};
    bf16 *dB_bf16{};
    bf16 *dC_bf16{};
    std::vector<bf16> hA_bf16_t;
    std::vector<bf16> hB_bf16_t;
    bf16 *dA_bf16_t{};
    bf16 *dB_bf16_t{};
    int num_iters;

    MatmulBuffers(int M, int K, int N) {
        this->M = M;
        this->K = K;
        this->N = N;
        hA = std::vector<float>(M * K);
        hB = std::vector<float>(K * N);
        hC = std::vector<float>(M * N);
        hA_bf16 = std::vector<bf16>(M * K);
        hB_bf16 = std::vector<bf16>(K * N);
        hC_bf16 = std::vector<bf16>(M * N);
        hA_bf16_t = std::vector<bf16>(K * M);
        hB_bf16_t = std::vector<bf16>(N * K);
        init();
    }

    void init() {
        // fill hA and hB with sequential numbers
        fillSequential(hA.data(), hA.size());
        fillSequential(hB.data(), hB.size());
        checkCuda(cudaMalloc(&dA, hA.size() * sizeof(float)), "cudaMalloc dA");
        checkCuda(cudaMalloc(&dB, hB.size() * sizeof(float)), "cudaMalloc dB");
        checkCuda(cudaMalloc(&dC, hC.size() * sizeof(float)), "cudaMalloc dC");
        checkCuda(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy hA->dA");
        checkCuda(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy hB->dB");
        // BF16 buffers
        fillSequential(hA_bf16.data(), hA_bf16.size());
        fillSequential(hB_bf16.data(), hB_bf16.size());
        checkCuda(cudaMalloc(&dA_bf16, hA_bf16.size() * sizeof(bf16)), "cudaMalloc dA_bf16");
        checkCuda(cudaMalloc(&dB_bf16, hB_bf16.size() * sizeof(bf16)), "cudaMalloc dB_bf16");
        checkCuda(cudaMalloc(&dC_bf16, hC_bf16.size() * sizeof(bf16)), "cudaMalloc dC_bf16");
        checkCuda(cudaMemcpy(dA_bf16, hA_bf16.data(), hA_bf16.size() * sizeof(bf16), cudaMemcpyHostToDevice),
                  "cudaMemcpy hA_bf16->dA_bf16");
        checkCuda(cudaMemcpy(dB_bf16, hB_bf16.data(), hB_bf16.size() * sizeof(bf16), cudaMemcpyHostToDevice),
                  "cudaMemcpy hB_bf16->dB_bf16");
        // BF16 col-major buffers
        fillSequentialColMajor(hA_bf16_t.data(), this->M, this->K);
        fillSequentialColMajor(hB_bf16_t.data(), this->N, this->K);
        checkCuda(cudaMalloc(&dA_bf16_t, hA_bf16_t.size() * sizeof(bf16)), "cudaMalloc dA_bf16_t");
        checkCuda(cudaMalloc(&dB_bf16_t, hB_bf16_t.size() * sizeof(bf16)), "cudaMalloc dB_bf16_t");
        checkCuda(cudaMemcpy(dA_bf16_t, hA_bf16_t.data(), hA_bf16_t.size() * sizeof(bf16), cudaMemcpyHostToDevice),
                  "cudaMemcpy hA_bf16_t->dA_bf16_t");
        checkCuda(cudaMemcpy(dB_bf16_t, hB_bf16_t.data(), hB_bf16_t.size() * sizeof(bf16), cudaMemcpyHostToDevice),
                  "cudaMemcpy hB_bf16_t->dB_bf16_t");
        reset();
    }

    void reset() {
        checkCuda(cudaMemset(dC, 0, hC.size() * sizeof(float)), "cudaMemset dC");
        checkCuda(cudaMemset(dC_bf16, 0, hC_bf16.size() * sizeof(bf16)), "cudaMemset dC_bf16");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        num_iters = 0;
    }

    void printResult(bool use_bf16 = false) {
        copyResultVector(use_bf16);
        printMatrix(hC.data(), M, N, "C");
    }

    std::vector<float> copyResultVector(bool use_bf16 = false, bool use_col_major = false) {
        if (use_bf16) {
            checkCuda(cudaMemcpy(hC_bf16.data(), dC_bf16, hC_bf16.size() * sizeof(bf16), cudaMemcpyDeviceToHost),
                      "cudaMemcpy dC_bf16->hC_bf16");
            if (use_col_major) {
                for (int i = 0; i < M; ++i) {
                    for (int j = 0; j < N; ++j) {
                        hC[i * N + j] = __bfloat162float(hC_bf16[j * M + i]);
                    }
                }
            } else {
                for (int i = 0; i < hC_bf16.size(); ++i) {
                    hC[i] = __bfloat162float(hC_bf16[i]);
                }
            }
        } else {
            checkCuda(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy dC->hC");
        }
        return hC;
    }

    void printTFLOPS(float elapsed_ms, std::string kernel_name) {
        double n_flop_per_iter = (static_cast<double>(M) * K * N) * 2;
        double tflops = num_iters * n_flop_per_iter / (elapsed_ms / 1000.0f) / 1e12;
        std::printf("M=%d K=%d N=%d (%s): %.3f TFLOPS\n", M, K, N, kernel_name.c_str(), tflops);
    }

    ~MatmulBuffers() {
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        cudaFree(dA_bf16);
        cudaFree(dB_bf16);
        cudaFree(dC_bf16);
        cudaFree(dA_bf16_t);
        cudaFree(dB_bf16_t);
    }
};

class DeviceTimer {
  public:
    DeviceTimer() {
        checkCuda(cudaEventCreate(&start_), "cudaEventCreate start");
        checkCuda(cudaEventCreate(&stop_), "cudaEventCreate stop");
    }

    ~DeviceTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() { checkCuda(cudaEventRecord(start_), "cudaEventRecord start"); }

    float stop() {
        checkCuda(cudaEventRecord(stop_), "cudaEventRecord stop");
        checkCuda(cudaEventSynchronize(stop_), "cudaEventSynchronize stop");
        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
        return ms;
    }

  private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};
