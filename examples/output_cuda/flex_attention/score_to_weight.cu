// PTO Program: score_to_weight
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: score_to_weight
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 1,056 bytes (1.0 KB)
//   Total capacity (w/ reuse): 544 bytes (0.5 KB)
//   Reuse savings:            512 bytes (48.5%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_scores           8x8        f32       256   [  4,   6]           <- scores
//   row_sum              8x1        f32        32   [  1,   6]           -
//   scores               8x8        f32       256   [  0,   3]           -
//   shifted              8x8        f32       256   [  3,   4]           -
//   weights              8x8        f32       256   [  6,   7]           <- shifted
//
// BUFFER REUSE MAP:
//   exp_scores reuses buffer of scores
//   weights reuses buffer of shifted
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

__device__ float scores[8][8];
__device__ float row_sum[8][1];
__device__ float shifted[8][8];
__device__ float exp_scores[8][8];
__device__ float weights[8][8];

__global__ void score_to_weight_kernel(float* scores_mem, float* weights_mem) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (1 ops): scores=TLOAD(...)
    if (_row < 8 && _col < 8) {
        scores[_row][_col] = scores_mem[_row * 8 + _col];
    }

    // TROWSUM: row_sum = rowsum(scores)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += scores[_row][_c];
        row_sum[_row][0] = _sum;}

    // FUSED (1 ops): row_sum=TDIVS(...)
    if (_row < 8 && _col < 1) {
        row_sum[_row][_col] = row_sum[_row][_col] / 8.0f;
    }

    // FUSED (2 ops): shifted=TROWEXPANDSUB(...); exp_scores=TEXP(...)
    if (_row < 8 && _col < 8) {
        shifted[_row][_col] = scores[_row][_col] - row_sum[_row][0];
        exp_scores[_row][_col] = __expf(shifted[_row][_col]);
    }

    // TROWSUM: row_sum = rowsum(exp_scores)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += exp_scores[_row][_c];
        row_sum[_row][0] = _sum;}

    // FUSED (2 ops): weights=TROWEXPANDDIV(...); weights_mem=TSTORE(...)
    if (_row < 8 && _col < 8) {
        weights[_row][_col] = exp_scores[_row][_col] / row_sum[_row][0];
        weights_mem[_row * 8 + _col] = weights[_row][_col];
    }

}

void score_to_weight(float* scores_mem, float* weights_mem) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    score_to_weight_kernel<<<grid, block>>>(scores_mem, weights_mem);
    cudaDeviceSynchronize();
}