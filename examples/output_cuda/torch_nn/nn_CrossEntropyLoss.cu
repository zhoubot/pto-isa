// PTO Program: nn_CrossEntropyLoss
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_CrossEntropyLoss
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     11
//   Total capacity (no reuse): 1,640 bytes (1.6 KB)
//   Total capacity (w/ reuse): 840 bytes (0.8 KB)
//   Reuse savings:            800 bytes (48.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_pred             8x8        f32       256   [  2,   3]           -
//   log_softmax          8x8        f32       256   [  5,   6]           <- exp_pred
//   log_sum              8x1        f32        32   [  4,   5]           -
//   neg_weighted         8x8        f32       256   [  7,   8]           <- target
//   pred                 8x8        f32       256   [  0,   5]           -
//   result               1x1        f32         4   [ 10,  11]           -
//   row_sum              8x1        f32        32   [  8,   9]           <- sum_exp
//   sum_exp              8x1        f32        32   [  3,   4]           -
//   target               8x8        f32       256   [  1,   6]           -
//   total_sum            1x1        f32         4   [  9,  10]           -
//   weighted             8x8        f32       256   [  6,   7]           <- pred
//
// BUFFER REUSE MAP:
//   log_softmax reuses buffer of exp_pred
//   weighted reuses buffer of pred
//   neg_weighted reuses buffer of target
//   row_sum reuses buffer of sum_exp
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
__device__ float exp_pred[8][8];
__device__ float sum_exp[8][1];
__device__ float log_sum[8][1];
__device__ float log_softmax[8][8];
__device__ float weighted[8][8];
__device__ float neg_weighted[8][8];
__device__ float row_sum[8][1];
__device__ float total_sum[1][1];
__device__ float result[1][1];

__global__ void nn_CrossEntropyLoss_kernel(float* pred_mem, float* target_mem, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 5 loop overheads saved

    // FUSED (3 ops): pred=TLOAD(...); target=TLOAD(...); exp_pred=TEXP(...)
    if (_row < 8 && _col < 8) {
        pred[_row][_col] = pred_mem[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
        exp_pred[_row][_col] = __expf(pred[_row][_col]);
    }

    // TROWSUM: sum_exp = rowsum(exp_pred)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += exp_pred[_row][_c];
        sum_exp[_row][0] = _sum;}

    // FUSED (1 ops): log_sum=TLOG(...)
    if (_row < 8 && _col < 1) {
        log_sum[_row][_col] = __logf(sum_exp[_row][_col]);
    }

    // FUSED (3 ops): log_softmax=TROWEXPANDSUB(...); weighted=TMUL(...); neg_weighted=TNEG(...)
    if (_row < 8 && _col < 8) {
        log_softmax[_row][_col] = pred[_row][_col] - log_sum[_row][0];
        weighted[_row][_col] = target[_row][_col] * log_softmax[_row][_col];
        neg_weighted[_row][_col] = -weighted[_row][_col];
    }

    // TROWSUM: row_sum = rowsum(neg_weighted)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += neg_weighted[_row][_c];
        row_sum[_row][0] = _sum;}

    // TCOLSUM: Not implemented

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = total_sum[_row][_col] / 8.0f;
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void nn_CrossEntropyLoss(float* pred_mem, float* target_mem, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_CrossEntropyLoss_kernel<<<grid, block>>>(pred_mem, target_mem, output);
    cudaDeviceSynchronize();
}