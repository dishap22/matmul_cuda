#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

void matrixMultiplyCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < N; ++col) {
            float value = 0;
            for (int i = 0; i < K; ++i)
                value += A[row * K + i] * B[i * N + col];
            C[row * N + col] = value;
        }
}

__global__ void matrixMultiplyNaive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0;
        for (int i = 0; i < K; ++i)
            value += A[row * K + i] * B[i * N + col];
        C[row * N + col] = value;
    }
}

__global__ void matrixMultiplyTiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float value = 0;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < M && t * TILE_WIDTH + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0;

        if (t * TILE_WIDTH + threadIdx.y < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = value;
}

namespace solution {
    std::string compute(const std::string &m1_path, const std::string &m2_path, int M, int K, int N) {
        auto h_A = std::make_unique<float[]>(M * K);
        auto h_B = std::make_unique<float[]>(K * N);
        auto h_C_cpu = std::make_unique<float[]>(M * N);
        auto h_C_naive = std::make_unique<float[]>(M * N);
        auto h_C_tiled = std::make_unique<float[]>(M * N);

        std::ifstream(m1_path).read(reinterpret_cast<char*>(h_A.get()), sizeof(float) * M * K);
        std::ifstream(m2_path).read(reinterpret_cast<char*>(h_B.get()), sizeof(float) * K * N);

        auto t1 = std::chrono::high_resolution_clock::now();
        matrixMultiplyCPU(h_A.get(), h_B.get(), h_C_cpu.get(), M, N, K);
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double>(t2 - t1).count();

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, sizeof(float) * M * K);
        cudaMalloc(&d_B, sizeof(float) * K * N);
        cudaMalloc(&d_C, sizeof(float) * M * N);
        cudaMemcpy(d_A, h_A.get(), sizeof(float) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.get(), sizeof(float) * K * N, cudaMemcpyHostToDevice);

        cudaMemset(d_C, 0, sizeof(float) * M * N);
        dim3 blockDim(32, 32);
        dim3 gridDim((N + 31) / 32, (M + 31) / 32);
        cudaEvent_t start, stop;

        float ms_naive;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        matrixMultiplyNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_naive, start, stop);
        cudaMemcpy(h_C_naive.get(), d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

        float ms_tiled;
        cudaMemset(d_C, 0, sizeof(float) * M * N);
        cudaEventRecord(start);
        matrixMultiplyTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_tiled, start, stop);
        cudaMemcpy(h_C_tiled.get(), d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(start); cudaEventDestroy(stop);

        double ops = 2.0 * M * N * K;
        double gflops_cpu = ops / (cpu_time * 1e9);
        double gflops_naive = ops / (ms_naive / 1e3 * 1e9);
        double gflops_tiled = ops / (ms_tiled / 1e3 * 1e9);

        std::cout << "CPU Time:   " << cpu_time << " sec, GFLOPS: " << gflops_cpu << std::endl;
        std::cout << "Naive GPU:  " << ms_naive / 1e3 << " sec, GFLOPS: " << gflops_naive << std::endl;
        std::cout << "Tiled GPU:  " << ms_tiled / 1e3 << " sec, GFLOPS: " << gflops_tiled << std::endl;

        std::string sol_path = std::filesystem::temp_directory_path() / "solution.dat";
        std::ofstream sol_file(sol_path, std::ios::binary);
        sol_file.write(reinterpret_cast<const char*>(h_C_tiled.get()), sizeof(float) * M * N);
        sol_file.close();
        return sol_path;
    }
}
