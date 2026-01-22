// PTO Program: F_pairwise_distance
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_pairwise_distance
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 1,088 bytes (1.1 KB)
//   Total capacity (w/ reuse): 832 bytes (0.8 KB)
//   Reuse savings:            256 bytes (23.5%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   diff                 8x8        f32       256   [  2,   3]           -
//   result               8x1        f32        32   [  5,   6]           -
//   row_sum              8x1        f32        32   [  4,   5]           -
//   sq_diff              8x8        f32       256   [  3,   4]           <- x1
//   x1                   8x8        f32       256   [  0,   2]           -
//   x2                   8x8        f32       256   [  1,   2]           -
//
// BUFFER REUSE MAP:
//   sq_diff reuses buffer of x1
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

__device__ float x1[8][8];
__device__ float x2[8][8];
__device__ float diff[8][8];
__device__ float sq_diff[8][8];
__device__ float row_sum[8][1];
__device__ float result[8][1];

__global__ void F_pairwise_distance_kernel(float* input1, float* input2, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (4 ops): x1=TLOAD(...); x2=TLOAD(...); diff=TSUB(...); sq_diff=TMUL(...)
    if (_row < 8 && _col < 8) {
        x1[_row][_col] = input1[_row * 8 + _col];
        x2[_row][_col] = input2[_row * 8 + _col];
        diff[_row][_col] = x1[_row][_col] - x2[_row][_col];
        sq_diff[_row][_col] = diff[_row][_col] * diff;
    }

    // TROWSUM: row_sum = rowsum(sq_diff)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += sq_diff[_row][_c];
        row_sum[_row][0] = _sum;}

    // FUSED (2 ops): result=TSQRT(...); output=TSTORE(...)
    if (_row < 8 && _col < 1) {
        result[_row][_col] = __fsqrt_rn(row_sum[_row][_col]);
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void F_pairwise_distance(float* input1, float* input2, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_pairwise_distance_kernel<<<grid, block>>>(input1, input2, output);
    cudaDeviceSynchronize();
}