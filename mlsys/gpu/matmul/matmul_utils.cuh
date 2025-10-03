#pragma once

#include <cstdlib>
#include <cstdio>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <string>

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

inline dim3 makeGrid2D(int rows, int cols, dim3 block) {
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
    int num_iters;

    MatmulBuffers(int M, int K, int N) {
        this->M = M;
        this->K = K;
        this->N = N;
        hA = std::vector<float>(M * K);
        hB = std::vector<float>(K * N);
        hC = std::vector<float>(M * N);
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
        reset();
    }

    void reset() {
        checkCuda(cudaMemset(dC, 0, hC.size() * sizeof(float)), "cudaMemset dC");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        num_iters = 0;
    }

    void printResult() {
        checkCuda(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy dC->hC");
        printMatrix(hC.data(), M, N, "C");
    }

    std::vector<float> copyResultVector() {
        checkCuda(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy dC->hC");
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
