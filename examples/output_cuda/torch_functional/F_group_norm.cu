// PTO Program: F_group_norm
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_group_norm
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 1,120 bytes (1.1 KB)
//   Total capacity (w/ reuse): 576 bytes (0.6 KB)
//   Reuse savings:            544 bytes (48.6%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   centered             8x8        f32       256   [  3,   9]           -
//   mean                 8x1        f32        32   [  1,   3]           -
//   result               8x8        f32       256   [  9,  10]           <- sq_centered
//   sq_centered          8x8        f32       256   [  4,   5]           <- x
//   std                  8x1        f32        32   [  8,   9]           -
//   var                  8x1        f32        32   [  5,   8]           <- mean
//   x                    8x8        f32       256   [  0,   3]           -
//
// BUFFER REUSE MAP:
//   sq_centered reuses buffer of x
//   var reuses buffer of mean
//   result reuses buffer of sq_centered
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
__device__ float mean[8][1];
__device__ float centered[8][8];
__device__ float sq_centered[8][8];
__device__ float var[8][1];
__device__ float std[8][1];
__device__ float result[8][8];

__global__ void F_group_norm_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
    }

    // TROWSUM: mean = rowsum(x)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += x[_row][_c];
        mean[_row][0] = _sum;}

    // FUSED (1 ops): mean=TDIVS(...)
    if (_row < 8 && _col < 1) {
        mean[_row][_col] = mean[_row][_col] / 8.0f;
    }

    // FUSED (2 ops): centered=TROWEXPANDSUB(...); sq_centered=TMUL(...)
    if (_row < 8 && _col < 8) {
        centered[_row][_col] = x[_row][_col] - mean[_row][0];
        sq_centered[_row][_col] = centered[_row][_col] * centered[_row][_col];
    }

    // TROWSUM: var = rowsum(sq_centered)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += sq_centered[_row][_c];
        var[_row][0] = _sum;}

    // FUSED (3 ops): var=TDIVS(...); var=TADDS(...); std=TSQRT(...)
    if (_row < 8 && _col < 1) {
        var[_row][_col] = var[_row][_col] / 8.0f;
        var[_row][_col] = var[_row][_col] + 1e-05f;
        std[_row][_col] = __fsqrt_rn(var[_row][_col]);
    }

    // FUSED (2 ops): result=TROWEXPANDDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        result[_row][_col] = centered[_row][_col] / std[_row][0];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_group_norm(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_group_norm_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}