// PTO Program: F_smooth_l1_loss
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_smooth_l1_loss
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     11
//   Total capacity (no reuse): 2,340 bytes (2.3 KB)
//   Total capacity (w/ reuse): 1,060 bytes (1.0 KB)
//   Reuse savings:            1,280 bytes (54.7%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   abs_diff             8x8        f32       256   [  3,   6]           <- pred
//   beta_tile            8x8        f32       256   [-, -]               -
//   diff                 8x8        f32       256   [  2,   4]           -
//   l1_part              8x8        f32       256   [  6,   7]           <- sq_diff
//   l2_part              8x8        f32       256   [  5,   7]           <- diff
//   loss                 8x8        f32       256   [  7,   8]           <- abs_diff
//   pred                 8x8        f32       256   [  0,   2]           -
//   result               1x1        f32         4   [  9,  11]           -
//   row_sum              8x1        f32        32   [  8,   9]           -
//   sq_diff              8x8        f32       256   [  4,   5]           <- target
//   target               8x8        f32       256   [  1,   2]           -
//
// BUFFER REUSE MAP:
//   abs_diff reuses buffer of pred
//   sq_diff reuses buffer of target
//   l2_part reuses buffer of diff
//   l1_part reuses buffer of sq_diff
//   loss reuses buffer of abs_diff
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
__device__ float sq_diff[8][8];
__device__ float l2_part[8][8];
__device__ float l1_part[8][8];
__device__ float beta_tile[8][8];
__device__ float loss[8][8];
__device__ float row_sum[8][1];
__device__ float result[1][1];

__global__ void F_smooth_l1_loss_kernel(float* input, float* target_mem, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 8 loop overheads saved

    // FUSED (8 ops): pred=TLOAD(...); target=TLOAD(...); diff=TSUB(...); abs_diff=TABS(...); sq_diff=TMUL(...); l2_part=TDIVS(...); l1_part=TADDS(...); loss=TMIN(...)
    if (_row < 8 && _col < 8) {
        pred[_row][_col] = input[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
        diff[_row][_col] = pred[_row][_col] - target[_row][_col];
        abs_diff[_row][_col] = fabsf(diff[_row][_col]);
        sq_diff[_row][_col] = diff[_row][_col] * diff;
        l2_part[_row][_col] = sq_diff[_row][_col] / 2.0f;
        l1_part[_row][_col] = abs_diff[_row][_col] + -0.5f;
        loss[_row][_col] = fminf(l2_part[_row][_col], l1_part[_row][_col]);
    }

    // TROWSUM: row_sum = rowsum(loss)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += loss[_row][_c];
        row_sum[_row][0] = _sum;}

    // TCOLSUM: Not implemented

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = result[_row][_col] / 64.0f;
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void F_smooth_l1_loss(float* input, float* target_mem, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_smooth_l1_loss_kernel<<<grid, block>>>(input, target_mem, output);
    cudaDeviceSynchronize();
}