[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/wvTvvz0E)

# GPU_Matmul Report

## Performance Summary

| Method        | Time (sec) | GFLOPS   | Speedup over CPU |
|---------------|------------|----------|------------------|
| **CPU**       | 48.5372    | 0.354    | 1×               |
| **Naive GPU** | 0.0281     | 610.814  | ~1,727×          |
| **Tiled GPU** | 0.0217     | 792.312  | ~2,236×          |

## Optimizations Made

- **Naive GPU**:
  - Used CUDA parallelization, where each thread is responsible for computing a single matrix element.
  - The parallelisation provided a **1,727× speedup** compared to the CPU implementation.
  - However this approach led to heavy use of global memory.

- **Tiled GPU**:
  - Implemented shared memory tiling for A and B sub-matrices.
  - This reduced global memory access and improved memory bandwidth efficiency.
  - It also minimized redundant loads through the data reuse in shared memory.
  - Used `__syncthreads()` to synchronize access within blocks.

## Observations

- Shared memory tiling provides a significant performance boost.
- Over **2,200× speedup** achieved compared to the CPU implementation and almost **1.3x** compared to the naive GPU version.
- High GFLOPS value demonstrates excellent GPU utilization.
- Performance difference grows with larger matrix sizes due to reduced global memory bottlenecks.

---
