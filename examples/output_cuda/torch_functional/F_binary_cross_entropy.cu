// PTO Program: F_binary_cross_entropy
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_binary_cross_entropy
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     11
//   Total capacity (no reuse): 2,340 bytes (2.3 KB)
//   Total capacity (w/ reuse): 1,316 bytes (1.3 KB)
//   Reuse savings:            1,024 bytes (43.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   bce                  8x8        f32       256   [ 10,  12]           <- log_pred
//   log_one_minus        8x8        f32       256   [  5,   9]           <- pred
//   log_pred             8x8        f32       256   [  2,   8]           -
//   one_minus_pred       8x8        f32       256   [  3,   5]           -
//   one_minus_target     8x8        f32       256   [  6,   9]           <- one_minus_pred
//   pred                 8x8        f32       256   [  0,   3]           -
//   result               1x1        f32         4   [ 13,  15]           -
//   row_sum              8x1        f32        32   [ 12,  13]           -
//   target               8x8        f32       256   [  1,   8]           -
//   term1                8x8        f32       256   [  8,  10]           -
//   term2                8x8        f32       256   [  9,  10]           <- target
//
// BUFFER REUSE MAP:
//   log_one_minus reuses buffer of pred
//   one_minus_target reuses buffer of one_minus_pred
//   term2 reuses buffer of target
//   bce reuses buffer of log_pred
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
__device__ float log_pred[8][8];
__device__ float one_minus_pred[8][8];
__device__ float log_one_minus[8][8];
__device__ float one_minus_target[8][8];
__device__ float term1[8][8];
__device__ float term2[8][8];
__device__ float bce[8][8];
__device__ float row_sum[8][1];
__device__ float result[1][1];

__global__ void F_binary_cross_entropy_kernel(float* input, float* target_mem, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 12 loop overheads saved

    // FUSED (12 ops): pred=TLOAD(...); target=TLOAD(...); log_pred=TLOG(...); one_minus_pred=TMULS(...); one_minus_pred=TADDS(...); log_one_minus=TLOG(...); one_minus_target=TMULS(...); one_minus_target=TADDS(...); term1=TMUL(...); term2=TMUL(...); bce=TADD(...); bce=TNEG(...)
    if (_row < 8 && _col < 8) {
        pred[_row][_col] = input[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
        log_pred[_row][_col] = __logf(pred[_row][_col]);
        one_minus_pred[_row][_col] = pred[_row][_col] * -1.0f;
        one_minus_pred[_row][_col] = one_minus_pred[_row][_col] + 1.0f;
        log_one_minus[_row][_col] = __logf(one_minus_pred[_row][_col]);
        one_minus_target[_row][_col] = target[_row][_col] * -1.0f;
        one_minus_target[_row][_col] = one_minus_target[_row][_col] + 1.0f;
        term1[_row][_col] = target[_row][_col] * log_pred[_row][_col];
        term2[_row][_col] = one_minus_target[_row][_col] * log_one_minus[_row][_col];
        bce[_row][_col] = term1[_row][_col] + term2[_row][_col];
        bce[_row][_col] = -bce[_row][_col];
    }

    // TROWSUM: row_sum = rowsum(bce)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += bce[_row][_c];
        row_sum[_row][0] = _sum;}

    // TCOLSUM: Not implemented

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = result[_row][_col] / 64.0f;
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void F_binary_cross_entropy(float* input, float* target_mem, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_binary_cross_entropy_kernel<<<grid, block>>>(input, target_mem, output);
    cudaDeviceSynchronize();
}