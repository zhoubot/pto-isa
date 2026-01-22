// PTO Program: F_hardtanh
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_hardtanh
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 1,280 bytes (1.2 KB)
//   Total capacity (w/ reuse): 1,024 bytes (1.0 KB)
//   Reuse savings:            256 bytes (20.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   clamp_low            8x8        f32       256   [  3,   4]           -
//   max_tile             8x8        f32       256   [  2,   4]           -
//   min_tile             8x8        f32       256   [  1,   3]           -
//   result               8x8        f32       256   [  4,   5]           <- x
//   x                    8x8        f32       256   [  0,   3]           -
//
// BUFFER REUSE MAP:
//   result reuses buffer of x
//
// ======================================================================

// Auto-generated CUDA code from PTO ISA Compiler
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

namespace cg = cooperative_groups;

__device__ float x[8][8];
__device__ float min_tile[8][8];
__device__ float max_tile[8][8];
__device__ float clamp_low[8][8];
__device__ float result[8][8];

__global__ void F_hardtanh_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 5 loop overheads saved

    // FUSED (6 ops): x=TLOAD(...); min_tile=TEXPANDS(...); max_tile=TEXPANDS(...); clamp_low=TMAX(...); result=TMIN(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        min_tile[_row][_col] = -1.0f;
        max_tile[_row][_col] = 1.0f;
        clamp_low[_row][_col] = fmaxf(x[_row][_col], min_tile[_row][_col]);
        result[_row][_col] = fminf(clamp_low[_row][_col], max_tile[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_hardtanh(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_hardtanh_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}