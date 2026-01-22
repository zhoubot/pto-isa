// PTO Program: nn_SmoothL1Loss
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_SmoothL1Loss
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     11
//   Total capacity (no reuse): 2,088 bytes (2.0 KB)
//   Total capacity (w/ reuse): 808 bytes (0.8 KB)
//   Reuse savings:            1,280 bytes (61.3%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   abs_diff             8x8        f32       256   [  3,   6]           <- pred
//   diff                 8x8        f32       256   [  2,   4]           -
//   l1_term              8x8        f32       256   [  6,   7]           <- squared
//   l2_term              8x8        f32       256   [  5,   7]           <- diff
//   pred                 8x8        f32       256   [  0,   2]           -
//   result               1x1        f32         4   [ 10,  11]           -
//   row_sum              8x1        f32        32   [  8,   9]           -
//   smooth               8x8        f32       256   [  7,   8]           <- abs_diff
//   squared              8x8        f32       256   [  4,   5]           <- target
//   target               8x8        f32       256   [  1,   2]           -
//   total_sum            1x1        f32         4   [  9,  10]           -
//
// BUFFER REUSE MAP:
//   abs_diff reuses buffer of pred
//   squared reuses buffer of target
//   l2_term reuses buffer of diff
//   l1_term reuses buffer of squared
//   smooth reuses buffer of abs_diff
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
__device__ float squared[8][8];
__device__ float l2_term[8][8];
__device__ float l1_term[8][8];
__device__ float smooth[8][8];
__device__ float row_sum[8][1];
__device__ float total_sum[1][1];
__device__ float result[1][1];

__global__ void nn_SmoothL1Loss_kernel(float* pred_mem, float* target_mem, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 8 loop overheads saved

    // FUSED (8 ops): pred=TLOAD(...); target=TLOAD(...); diff=TSUB(...); abs_diff=TABS(...); squared=TMUL(...); l2_term=TDIVS(...); l1_term=TADDS(...); smooth=TMIN(...)
    if (_row < 8 && _col < 8) {
        pred[_row][_col] = pred_mem[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
        diff[_row][_col] = pred[_row][_col] - target[_row][_col];
        abs_diff[_row][_col] = fabsf(diff[_row][_col]);
        squared[_row][_col] = diff[_row][_col] * diff;
        l2_term[_row][_col] = squared[_row][_col] / 2.0f;
        l1_term[_row][_col] = abs_diff[_row][_col] + -0.5f;
        smooth[_row][_col] = fminf(l2_term[_row][_col], l1_term[_row][_col]);
    }

    // TROWSUM: row_sum = rowsum(smooth)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += smooth[_row][_c];
        row_sum[_row][0] = _sum;}

    // TCOLSUM: Not implemented

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = total_sum[_row][_col] / 64.0f;
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void nn_SmoothL1Loss(float* pred_mem, float* target_mem, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_SmoothL1Loss_kernel<<<grid, block>>>(pred_mem, target_mem, output);
    cudaDeviceSynchronize();
}