// PTO Program: nn_L1Loss
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_L1Loss
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 1,064 bytes (1.0 KB)
//   Total capacity (w/ reuse): 808 bytes (0.8 KB)
//   Reuse savings:            256 bytes (24.1%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   abs_diff             8x8        f32       256   [  3,   4]           <- pred
//   diff                 8x8        f32       256   [  2,   3]           -
//   pred                 8x8        f32       256   [  0,   2]           -
//   result               1x1        f32         4   [  6,   7]           -
//   row_sum              8x1        f32        32   [  4,   5]           -
//   target               8x8        f32       256   [  1,   2]           -
//   total_sum            1x1        f32         4   [  5,   6]           -
//
// BUFFER REUSE MAP:
//   abs_diff reuses buffer of pred
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

__device__ float pred[8][8];
__device__ float target[8][8];
__device__ float diff[8][8];
__device__ float abs_diff[8][8];
__device__ float row_sum[8][1];
__device__ float total_sum[1][1];
__device__ float result[1][1];

__global__ void nn_L1Loss_kernel(float* pred_mem, float* target_mem, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (4 ops): pred=TLOAD(...); target=TLOAD(...); diff=TSUB(...); abs_diff=TABS(...)
    if (_row < 8 && _col < 8) {
        pred[_row][_col] = pred_mem[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
        diff[_row][_col] = pred[_row][_col] - target[_row][_col];
        abs_diff[_row][_col] = fabsf(diff[_row][_col]);
    }

    // TROWSUM: row_sum = rowsum(abs_diff)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += abs_diff[_row][_c];
        row_sum[_row][0] = _sum;}

    // TCOLSUM: Not implemented

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = total_sum[_row][_col] / 64.0f;
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void nn_L1Loss(float* pred_mem, float* target_mem, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_L1Loss_kernel<<<grid, block>>>(pred_mem, target_mem, output);
    cudaDeviceSynchronize();
}