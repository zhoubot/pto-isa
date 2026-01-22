// PTO Program: nn_RMSNorm
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_RMSNorm
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 896 bytes (0.9 KB)
//   Total capacity (w/ reuse): 576 bytes (0.6 KB)
//   Reuse savings:            320 bytes (35.7%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   mean_sq              8x1        f32        32   [  3,   4]           -
//   mean_sq_eps          8x1        f32        32   [  4,   5]           <- mean_sq_sum
//   mean_sq_sum          8x1        f32        32   [  2,   3]           -
//   result               8x8        f32       256   [  6,   7]           <- x_squared
//   rms                  8x1        f32        32   [  5,   6]           <- mean_sq
//   x                    8x8        f32       256   [  0,   6]           -
//   x_squared            8x8        f32       256   [  1,   2]           -
//
// BUFFER REUSE MAP:
//   mean_sq_eps reuses buffer of mean_sq_sum
//   rms reuses buffer of mean_sq
//   result reuses buffer of x_squared
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
__device__ float x_squared[8][8];
__device__ float mean_sq_sum[8][1];
__device__ float mean_sq[8][1];
__device__ float mean_sq_eps[8][1];
__device__ float rms[8][1];
__device__ float result[8][8];

__global__ void nn_RMSNorm_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (2 ops): x=TLOAD(...); x_squared=TMUL(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        x_squared[_row][_col] = x[_row][_col] * x[_row][_col];
    }

    // TROWSUM: mean_sq_sum = rowsum(x_squared)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += x_squared[_row][_c];
        mean_sq_sum[_row][0] = _sum;}

    // FUSED (3 ops): mean_sq=TDIVS(...); mean_sq_eps=TADDS(...); rms=TSQRT(...)
    if (_row < 8 && _col < 1) {
        mean_sq[_row][_col] = mean_sq_sum[_row][_col] / 8.0f;
        mean_sq_eps[_row][_col] = mean_sq[_row][_col] + 1e-05f;
        rms[_row][_col] = __fsqrt_rn(mean_sq_eps[_row][_col]);
    }

    // FUSED (2 ops): result=TDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        result[_row][_col] = x[_row][_col] / rms[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_RMSNorm(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_RMSNorm_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}