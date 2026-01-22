// PTO Program: F_batch_norm
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_batch_norm
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 864 bytes (0.8 KB)
//   Total capacity (w/ reuse): 864 bytes (0.8 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   centered             8x8        f32       256   [  3,  -1]           -
//   mean                 1x8        f32        32   [  1,  -1]           -
//   result               8x8        f32       256   [  6,   7]           -
//   std                  1x8        f32        32   [  5,   6]           -
//   var                  1x8        f32        32   [  2,   5]           -
//   x                    8x8        f32       256   [  0,   6]           -
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
__device__ float mean[1][8];
__device__ float var[1][8];
__device__ float std[1][8];
__device__ float centered[8][8];
__device__ float result[8][8];

__global__ void F_batch_norm_kernel(float* input, float* mean_mem, float* var_mem, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
    }

    // FUSED (2 ops): mean=TLOAD(...); var=TLOAD(...)
    if (_row < 1 && _col < 8) {
        mean[_row][_col] = mean_mem[_row * 8 + _col];
        var[_row][_col] = var_mem[_row * 8 + _col];
    }

    // TCOLSUM: Not implemented

    // FUSED (2 ops): var=TADDS(...); std=TSQRT(...)
    if (_row < 1 && _col < 8) {
        var[_row][_col] = var[_row][_col] + 1e-05f;
        std[_row][_col] = __fsqrt_rn(var[_row][_col]);
    }

    // FUSED (2 ops): result=TDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        result[_row][_col] = x[_row][_col] / std[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_batch_norm(float* input, float* mean_mem, float* var_mem, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_batch_norm_kernel<<<grid, block>>>(input, mean_mem, var_mem, output);
    cudaDeviceSynchronize();
}