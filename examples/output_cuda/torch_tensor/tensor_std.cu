// PTO Program: tensor_std
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_std
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     10
//   Total capacity (no reuse): 1,104 bytes (1.1 KB)
//   Total capacity (w/ reuse): 808 bytes (0.8 KB)
//   Reuse savings:            296 bytes (26.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   centered             8x8        f32       256   [  5,   6]           -
//   mean_val             8x8        f32       256   [  4,   5]           -
//   result               1x1        f32         4   [ 10,  11]           <- var_total
//   row_sum              8x1        f32        32   [  1,   2]           -
//   self                 8x8        f32       256   [  0,   5]           -
//   sq_centered          8x8        f32       256   [  6,   7]           <- self
//   sq_row_sum           8x1        f32        32   [  7,   8]           <- row_sum
//   total                1x1        f32         4   [  2,   3]           -
//   var                  1x1        f32         4   [  9,  10]           -
//   var_total            1x1        f32         4   [  8,   9]           <- total
//
// BUFFER REUSE MAP:
//   sq_centered reuses buffer of self
//   sq_row_sum reuses buffer of row_sum
//   var_total reuses buffer of total
//   result reuses buffer of var_total
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

__device__ float self[8][8];
__device__ float row_sum[8][1];
__device__ float total[1][1];
__device__ float mean_val[8][8];
__device__ float centered[8][8];
__device__ float sq_centered[8][8];
__device__ float sq_row_sum[8][1];
__device__ float var_total[1][1];
__device__ float var[1][1];
__device__ float result[1][1];

__global__ void tensor_std_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (1 ops): self=TLOAD(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input[_row * 8 + _col];
    }

    // TROWSUM: row_sum = rowsum(self)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += self[_row][_c];
        row_sum[_row][0] = _sum;}

    // TCOLSUM: Not implemented

    // FUSED (1 ops): total=TDIVS(...)
    if (_row < 1 && _col < 1) {
        total[_row][_col] = total[_row][_col] / 64.0f;
    }

    // FUSED (3 ops): mean_val=TEXPANDS(...); centered=TSUB(...); sq_centered=TMUL(...)
    if (_row < 8 && _col < 8) {
        mean_val[_row][_col] = 0.0f;
        centered[_row][_col] = self[_row][_col] - mean_val[_row][_col];
        sq_centered[_row][_col] = centered[_row][_col] * centered[_row][_col];
    }

    // TROWSUM: sq_row_sum = rowsum(sq_centered)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += sq_centered[_row][_c];
        sq_row_sum[_row][0] = _sum;}

    // TCOLSUM: Not implemented

    // FUSED (3 ops): var=TDIVS(...); result=TSQRT(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        var[_row][_col] = var_total[_row][_col] / 64.0f;
        result[_row][_col] = __fsqrt_rn(var[_row][_col]);
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void tensor_std(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_std_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}