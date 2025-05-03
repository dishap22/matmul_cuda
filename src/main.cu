#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

// Simple matrix multiplication kernel
__global__ void matmulKernel(const float* A, const float* B, float* C, int n, int k, int m) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float value = 0.0f;

    for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < n && t * TILE_WIDTH + threadIdx.x < k)
            tileA[threadIdx.y][threadIdx.x] = A[row * k + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_WIDTH + threadIdx.y < k && col < m)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * m + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < m)
        C[row * m + col] = value;
}

namespace solution {
    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        // Create output file
        std::string sol_path = std::filesystem::temp_directory_path() / "solution.dat";
        std::ofstream sol_file(sol_path, std::ios::binary);

        // Load input matrices
        auto h_A = std::make_unique<float[]>(n*k);
        auto h_B = std::make_unique<float[]>(k*m);
        auto h_C = std::make_unique<float[]>(n*m);

        std::ifstream(m1_path).read(reinterpret_cast<char*>(h_A.get()), sizeof(float)*n*k);
        std::ifstream(m2_path).read(reinterpret_cast<char*>(h_B.get()), sizeof(float)*k*m);

        // GPU memory pointers
        float *d_A, *d_B, *d_C;

        // Allocate GPU memory
        cudaMalloc((void**)&d_A, sizeof(float)*n*k);
        cudaMalloc((void**)&d_B, sizeof(float)*k*m);
        cudaMalloc((void**)&d_C, sizeof(float)*n*m);

        // Copy data to GPU
        cudaMemcpy(d_A, h_A.get(), sizeof(float)*n*k, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.get(), sizeof(float)*k*m, cudaMemcpyHostToDevice);

        // Launch kernel (basic configuration)
        dim3 Db(TILE_WIDTH, TILE_WIDTH);
        dim3 Dg((m + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH);
        size_t Ns = 0;
        cudaStream_t S = 0;

        matmulKernel<<< Dg, Db, Ns, S >>>(d_A, d_B, d_C, n, k, m);

        // Copy result back
        cudaMemcpy(h_C.get(), d_C, sizeof(float)*n*m, cudaMemcpyDeviceToHost);

        // GPU Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // Save result
        sol_file.write(reinterpret_cast<const char*>(h_C.get()), sizeof(float)*n*m);
        sol_file.close();
        return sol_path;
    }
}
