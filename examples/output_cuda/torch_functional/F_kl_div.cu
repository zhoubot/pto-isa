// PTO Program: F_kl_div
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_kl_div
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 1,316 bytes (1.3 KB)
//   Total capacity (w/ reuse): 1,060 bytes (1.0 KB)
//   Reuse savings:            256 bytes (19.5%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   diff                 8x8        f32       256   [  3,   4]           -
//   kl                   8x8        f32       256   [  4,   5]           <- log_pred
//   log_pred             8x8        f32       256   [  0,   3]           -
//   log_target           8x8        f32       256   [  2,   3]           -
//   result               1x1        f32         4   [  6,   8]           -
//   row_sum              8x1        f32        32   [  5,   6]           -
//   target               8x8        f32       256   [  1,   4]           -
//
// BUFFER REUSE MAP:
//   kl reuses buffer of log_pred
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

__device__ float log_pred[8][8];
__device__ float target[8][8];
__device__ float log_target[8][8];
__device__ float diff[8][8];
__device__ float kl[8][8];
__device__ float row_sum[8][1];
__device__ float result[1][1];

__global__ void F_kl_div_kernel(float* input, float* target_mem, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 5 loop overheads saved

    // FUSED (5 ops): log_pred=TLOAD(...); target=TLOAD(...); log_target=TLOG(...); diff=TSUB(...); kl=TMUL(...)
    if (_row < 8 && _col < 8) {
        log_pred[_row][_col] = input[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
        log_target[_row][_col] = __logf(target[_row][_col]);
        diff[_row][_col] = log_target[_row][_col] - log_pred[_row][_col];
        kl[_row][_col] = target[_row][_col] * diff;
    }

    // TROWSUM: row_sum = rowsum(kl)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += kl[_row][_c];
        row_sum[_row][0] = _sum;}

    // TCOLSUM: Not implemented

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = result[_row][_col] / 64.0f;
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void F_kl_div(float* input, float* target_mem, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_kl_div_kernel<<<grid, block>>>(input, target_mem, output);
    cudaDeviceSynchronize();
}