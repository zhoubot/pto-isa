// PTO Program: F_cross_entropy
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_cross_entropy
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     11
//   Total capacity (no reuse): 1,668 bytes (1.6 KB)
//   Total capacity (w/ reuse): 836 bytes (0.8 KB)
//   Reuse savings:            832 bytes (49.9%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   ce                   8x8        f32       256   [  9,  11]           <- shifted
//   ce_row               8x1        f32        32   [ 11,  12]           <- row_sum
//   exp_shifted          8x8        f32       256   [  5,   6]           <- logits
//   log_softmax          8x8        f32       256   [  8,   9]           <- exp_shifted
//   log_sum              8x1        f32        32   [  7,   8]           -
//   logits               8x8        f32       256   [  0,   4]           -
//   result               1x1        f32         4   [ 12,  14]           -
//   row_mean             8x1        f32        32   [  2,   4]           -
//   row_sum              8x1        f32        32   [  6,   7]           <- row_mean
//   shifted              8x8        f32       256   [  4,   8]           -
//   target               8x8        f32       256   [  1,   9]           -
//
// BUFFER REUSE MAP:
//   exp_shifted reuses buffer of logits
//   row_sum reuses buffer of row_mean
//   log_softmax reuses buffer of exp_shifted
//   ce reuses buffer of shifted
//   ce_row reuses buffer of row_sum
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

__device__ float logits[8][8];
__device__ float target[8][8];
__device__ float row_mean[8][1];
__device__ float shifted[8][8];
__device__ float exp_shifted[8][8];
__device__ float row_sum[8][1];
__device__ float log_sum[8][1];
__device__ float log_softmax[8][8];
__device__ float ce[8][8];
__device__ float ce_row[8][1];
__device__ float result[1][1];

__global__ void F_cross_entropy_kernel(float* input, float* target_mem, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 5 loop overheads saved

    // FUSED (2 ops): logits=TLOAD(...); target=TLOAD(...)
    if (_row < 8 && _col < 8) {
        logits[_row][_col] = input[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
    }

    // TROWSUM: row_mean = rowsum(logits)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += logits[_row][_c];
        row_mean[_row][0] = _sum;}

    // FUSED (1 ops): row_mean=TDIVS(...)
    if (_row < 8 && _col < 1) {
        row_mean[_row][_col] = row_mean[_row][_col] / 8.0f;
    }

    // FUSED (2 ops): shifted=TROWEXPANDSUB(...); exp_shifted=TEXP(...)
    if (_row < 8 && _col < 8) {
        shifted[_row][_col] = logits[_row][_col] - row_mean[_row][0];
        exp_shifted[_row][_col] = __expf(shifted[_row][_col]);
    }

    // TROWSUM: row_sum = rowsum(exp_shifted)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += exp_shifted[_row][_c];
        row_sum[_row][0] = _sum;}

    // FUSED (1 ops): log_sum=TLOG(...)
    if (_row < 8 && _col < 1) {
        log_sum[_row][_col] = __logf(row_sum[_row][_col]);
    }

    // FUSED (3 ops): log_softmax=TROWEXPANDSUB(...); ce=TMUL(...); ce=TNEG(...)
    if (_row < 8 && _col < 8) {
        log_softmax[_row][_col] = shifted[_row][_col] - log_sum[_row][0];
        ce[_row][_col] = target[_row][_col] * log_softmax[_row][_col];
        ce[_row][_col] = -ce[_row][_col];
    }

    // TROWSUM: ce_row = rowsum(ce)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += ce[_row][_c];
        ce_row[_row][0] = _sum;}

    // TCOLSUM: Not implemented

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = result[_row][_col] / 8.0f;
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void F_cross_entropy(float* input, float* target_mem, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_cross_entropy_kernel<<<grid, block>>>(input, target_mem, output);
    cudaDeviceSynchronize();
}