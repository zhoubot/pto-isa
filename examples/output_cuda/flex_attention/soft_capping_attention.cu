// PTO Program: soft_capping_attention
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: soft_capping_attention
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     17
//   Total capacity (no reuse): 4,128 bytes (4.0 KB)
//   Total capacity (w/ reuse): 1,568 bytes (1.5 KB)
//   Reuse savings:            2,560 bytes (62.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   K                    8x8        f32       256   [  1,  -1]           -
//   Q                    8x8        f32       256   [  0,  -1]           -
//   V                    8x8        f32       256   [  2,  -1]           -
//   attn                 8x8        f32       256   [ 17,  -1]           <- capped_scores
//   capped_scores        8x8        f32       256   [ 11,  14]           <- exp_minus_1
//   exp_2x               8x8        f32       256   [  7,   9]           <- x_div_cap
//   exp_minus_1          8x8        f32       256   [  8,  10]           <- two_x
//   exp_plus_1           8x8        f32       256   [  9,  10]           -
//   exp_scores           8x8        f32       256   [ 15,  17]           <- tanh_x
//   output               8x8        f32       256   [ 18,  19]           <- shifted
//   row_sum              8x1        f32        32   [ 12,  17]           -
//   scaled               8x8        f32       256   [  4,   5]           -
//   scores               8x8        f32       256   [  3,   4]           -
//   shifted              8x8        f32       256   [ 14,  15]           <- exp_plus_1
//   tanh_x               8x8        f32       256   [ 10,  11]           <- exp_2x
//   two_x                8x8        f32       256   [  6,   7]           <- scaled
//   x_div_cap            8x8        f32       256   [  5,   6]           <- scores
//
// BUFFER REUSE MAP:
//   x_div_cap reuses buffer of scores
//   two_x reuses buffer of scaled
//   exp_2x reuses buffer of x_div_cap
//   exp_minus_1 reuses buffer of two_x
//   tanh_x reuses buffer of exp_2x
//   capped_scores reuses buffer of exp_minus_1
//   shifted reuses buffer of exp_plus_1
//   exp_scores reuses buffer of tanh_x
//   attn reuses buffer of capped_scores
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
__device__ float x_div_cap[8][8];
__device__ float two_x[8][8];
__device__ float exp_2x[8][8];
__device__ float exp_minus_1[8][8];
__device__ float exp_plus_1[8][8];
__device__ float tanh_x[8][8];
__device__ float capped_scores[8][8];
__device__ float row_sum[8][1];
__device__ float shifted[8][8];
__device__ float exp_scores[8][8];
__device__ float attn[8][8];
__device__ float output[8][8];

__global__ void soft_capping_attention_kernel(float* Q_mem, float* K_mem, float* V_mem, float* output_mem) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 10 loop overheads saved

    // FUSED (3 ops): Q=TLOAD(...); K=TLOAD(...); V=TLOAD(...)
    if (_row < 8 && _col < 8) {
        Q[_row][_col] = Q_mem[_row * 8 + _col];
        K[_row][_col] = K_mem[_row * 8 + _col];
        V[_row][_col] = V_mem[_row * 8 + _col];
    }

    // TMATMUL: scores = Q @ K
    if (_row < 8 && _col < 8) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 8; _k++) _sum += Q[_row][_k] * K[_k][_col];
        scores[_row][_col] = _sum;}

    // FUSED (8 ops): scaled=TMULS(...); x_div_cap=TDIVS(...); two_x=TMULS(...); exp_2x=TEXP(...); exp_minus_1=TADDS(...); exp_plus_1=TADDS(...); tanh_x=TDIV(...); capped_scores=TMULS(...)
    if (_row < 8 && _col < 8) {
        scaled[_row][_col] = scores[_row][_col] * 0.35355339059327373f;
        x_div_cap[_row][_col] = scaled[_row][_col] / 50.0f;
        two_x[_row][_col] = x_div_cap[_row][_col] * 2.0f;
        exp_2x[_row][_col] = __expf(two_x[_row][_col]);
        exp_minus_1[_row][_col] = exp_2x[_row][_col] + -1.0f;
        exp_plus_1[_row][_col] = exp_2x[_row][_col] + 1.0f;
        tanh_x[_row][_col] = exp_minus_1[_row][_col] / exp_plus_1[_row][_col];
        capped_scores[_row][_col] = tanh_x[_row][_col] * 50.0f;
    }

    // TROWSUM: row_sum = rowsum(capped_scores)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += capped_scores[_row][_c];
        row_sum[_row][0] = _sum;}

    // FUSED (1 ops): row_sum=TDIVS(...)
    if (_row < 8 && _col < 1) {
        row_sum[_row][_col] = row_sum[_row][_col] / 8.0f;
    }

    // FUSED (2 ops): shifted=TROWEXPANDSUB(...); exp_scores=TEXP(...)
    if (_row < 8 && _col < 8) {
        shifted[_row][_col] = capped_scores[_row][_col] - row_sum[_row][0];
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

void soft_capping_attention(float* Q_mem, float* K_mem, float* V_mem, float* output_mem) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    soft_capping_attention_kernel<<<grid, block>>>(Q_mem, K_mem, V_mem, output_mem);
    cudaDeviceSynchronize();
}