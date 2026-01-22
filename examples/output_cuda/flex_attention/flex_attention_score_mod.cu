// PTO Program: flex_attention_score_mod
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flex_attention_score_mod
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     12
//   Total capacity (no reuse): 2,848 bytes (2.8 KB)
//   Total capacity (w/ reuse): 1,568 bytes (1.5 KB)
//   Reuse savings:            1,280 bytes (44.9%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   K                    8x8        f32       256   [  1,  -1]           -
//   Q                    8x8        f32       256   [  0,  -1]           -
//   V                    8x8        f32       256   [  2,  -1]           -
//   attn                 8x8        f32       256   [ 12,  -1]           <- modified_scores
//   exp_scores           8x8        f32       256   [ 10,  12]           <- scaled
//   modified_scores      8x8        f32       256   [  6,   9]           <- scores
//   output               8x8        f32       256   [ 13,  14]           <- shifted
//   row_sum              8x1        f32        32   [  7,  12]           -
//   scaled               8x8        f32       256   [  5,   6]           -
//   score_mod_bias       8x8        f32       256   [  3,   6]           -
//   scores               8x8        f32       256   [  4,   5]           -
//   shifted              8x8        f32       256   [  9,  10]           <- score_mod_bias
//
// BUFFER REUSE MAP:
//   modified_scores reuses buffer of scores
//   shifted reuses buffer of score_mod_bias
//   exp_scores reuses buffer of scaled
//   attn reuses buffer of modified_scores
//   output reuses buffer of shifted
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

__device__ float Q[8][8];
__device__ float K[8][8];
__device__ float V[8][8];
__device__ float scores[8][8];
__device__ float scaled[8][8];
__device__ float score_mod_bias[8][8];
__device__ float modified_scores[8][8];
__device__ float row_sum[8][1];
__device__ float shifted[8][8];
__device__ float exp_scores[8][8];
__device__ float attn[8][8];
__device__ float output[8][8];

__global__ void flex_attention_score_mod_kernel(float* Q_mem, float* K_mem, float* V_mem, float* score_mod_mem, float* output_mem) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 5 loop overheads saved

    // FUSED (4 ops): Q=TLOAD(...); K=TLOAD(...); V=TLOAD(...); score_mod_bias=TLOAD(...)
    if (_row < 8 && _col < 8) {
        Q[_row][_col] = Q_mem[_row * 8 + _col];
        K[_row][_col] = K_mem[_row * 8 + _col];
        V[_row][_col] = V_mem[_row * 8 + _col];
        score_mod_bias[_row][_col] = score_mod_mem[_row * 8 + _col];
    }

    // TMATMUL: scores = Q @ K
    if (_row < 8 && _col < 8) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 8; _k++) _sum += Q[_row][_k] * K[_k][_col];
        scores[_row][_col] = _sum;}

    // FUSED (2 ops): scaled=TMULS(...); modified_scores=TADD(...)
    if (_row < 8 && _col < 8) {
        scaled[_row][_col] = scores[_row][_col] * 0.35355339059327373f;
        modified_scores[_row][_col] = scaled[_row][_col] + score_mod_bias[_row][_col];
    }

    // TROWSUM: row_sum = rowsum(modified_scores)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += modified_scores[_row][_c];
        row_sum[_row][0] = _sum;}

    // FUSED (1 ops): row_sum=TDIVS(...)
    if (_row < 8 && _col < 1) {
        row_sum[_row][_col] = row_sum[_row][_col] / 8.0f;
    }

    // FUSED (2 ops): shifted=TROWEXPANDSUB(...); exp_scores=TEXP(...)
    if (_row < 8 && _col < 8) {
        shifted[_row][_col] = modified_scores[_row][_col] - row_sum[_row][0];
        exp_scores[_row][_col] = __expf(shifted[_row][_col]);
    }

    // TROWSUM: row_sum = rowsum(exp_scores)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += exp_scores[_row][_c];
        row_sum[_row][0] = _sum;}

    // FUSED (1 ops): attn=TROWEXPANDDIV(...)
    if (_row < 8 && _col < 8) {
        attn[_row][_col] = exp_scores[_row][_col] / row_sum[_row][0];
    }

    // TMATMUL: output = attn @ V
    if (_row < 8 && _col < 8) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 8; _k++) _sum += attn[_row][_k] * V[_k][_col];
        output[_row][_col] = _sum;}

    // FUSED (1 ops): output_mem=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output_mem[_row * 8 + _col] = output[_row][_col];
    }

}

void flex_attention_score_mod(float* Q_mem, float* K_mem, float* V_mem, float* score_mod_mem, float* output_mem) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    flex_attention_score_mod_kernel<<<grid, block>>>(Q_mem, K_mem, V_mem, score_mod_mem, output_mem);
    cudaDeviceSynchronize();
}