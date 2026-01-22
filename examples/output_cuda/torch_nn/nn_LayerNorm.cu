// PTO Program: nn_LayerNorm
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_LayerNorm
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     10
//   Total capacity (no reuse): 1,216 bytes (1.2 KB)
//   Total capacity (w/ reuse): 576 bytes (0.6 KB)
//   Reuse savings:            640 bytes (52.6%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   mean                 8x1        f32        32   [  2,   3]           -
//   result               8x8        f32       256   [  9,  10]           <- squared
//   row_sum              8x1        f32        32   [  1,   2]           -
//   squared              8x8        f32       256   [  4,   5]           <- x
//   std                  8x1        f32        32   [  8,   9]           <- variance
//   var_eps              8x1        f32        32   [  7,   8]           <- var_sum
//   var_sum              8x1        f32        32   [  5,   6]           <- row_sum
//   variance             8x1        f32        32   [  6,   7]           <- mean
//   x                    8x8        f32       256   [  0,   3]           -
//   x_minus_mean         8x8        f32       256   [  3,   9]           -
//
// BUFFER REUSE MAP:
//   squared reuses buffer of x
//   var_sum reuses buffer of row_sum
//   variance reuses buffer of mean
//   var_eps reuses buffer of var_sum
//   std reuses buffer of variance
//   result reuses buffer of squared
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
__device__ float row_sum[8][1];
__device__ float mean[8][1];
__device__ float x_minus_mean[8][8];
__device__ float squared[8][8];
__device__ float var_sum[8][1];
__device__ float variance[8][1];
__device__ float var_eps[8][1];
__device__ float std[8][1];
__device__ float result[8][8];

__global__ void nn_LayerNorm_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
    }

    // TROWSUM: row_sum = rowsum(x)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += x[_row][_c];
        row_sum[_row][0] = _sum;}

    // FUSED (1 ops): mean=TDIVS(...)
    if (_row < 8 && _col < 1) {
        mean[_row][_col] = row_sum[_row][_col] / 8.0f;
    }

    // FUSED (2 ops): x_minus_mean=TROWEXPANDSUB(...); squared=TMUL(...)
    if (_row < 8 && _col < 8) {
        x_minus_mean[_row][_col] = x[_row][_col] - mean[_row][0];
        squared[_row][_col] = x_minus_mean[_row][_col] * x_minus_mean[_row][_col];
    }

    // TROWSUM: var_sum = rowsum(squared)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += squared[_row][_c];
        var_sum[_row][0] = _sum;}

    // FUSED (3 ops): variance=TDIVS(...); var_eps=TADDS(...); std=TSQRT(...)
    if (_row < 8 && _col < 1) {
        variance[_row][_col] = var_sum[_row][_col] / 8.0f;
        var_eps[_row][_col] = variance[_row][_col] + 1e-05f;
        std[_row][_col] = __fsqrt_rn(var_eps[_row][_col]);
    }

    // FUSED (2 ops): result=TROWEXPANDDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        result[_row][_col] = x_minus_mean[_row][_col] / std[_row][0];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_LayerNorm(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_LayerNorm_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}