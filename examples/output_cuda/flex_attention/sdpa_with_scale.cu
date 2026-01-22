// PTO Program: sdpa_with_scale
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: sdpa_with_scale
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     10
//   Total capacity (no reuse): 2,336 bytes (2.3 KB)
//   Total capacity (w/ reuse): 1,312 bytes (1.3 KB)
//   Reuse savings:            1,024 bytes (43.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   K                    8x8        f32       256   [  1,  -1]           -
//   Q                    8x8        f32       256   [  0,  -1]           -
//   V                    8x8        f32       256   [  2,  -1]           -
//   attn                 8x8        f32       256   [ 10,  -1]           <- shifted
//   exp_scores           8x8        f32       256   [  8,  10]           <- scaled
//   output               8x8        f32       256   [ 11,  12]           <- exp_scores
//   row_sum              8x1        f32        32   [  5,  10]           -
//   scaled               8x8        f32       256   [  4,   7]           -
//   scores               8x8        f32       256   [  3,   4]           -
//   shifted              8x8        f32       256   [  7,   8]           <- scores
//
// BUFFER REUSE MAP:
//   shifted reuses buffer of scores
//   exp_scores reuses buffer of scaled
//   attn reuses buffer of shifted
//   output reuses buffer of exp_scores
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
__device__ float row_sum[8][1];
__device__ float shifted[8][8];
__device__ float exp_scores[8][8];
__device__ float attn[8][8];
__device__ float output[8][8];

__global__ void sdpa_with_scale_kernel(float* Q_mem, float* K_mem, float* V_mem, float* output_mem) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

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

    // FUSED (1 ops): scaled=TMULS(...)
    if (_row < 8 && _col < 8) {
        scaled[_row][_col] = scores[_row][_col] * 0.35355339059327373f;
    }

    // TROWSUM: row_sum = rowsum(scaled)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += scaled[_row][_c];
        row_sum[_row][0] = _sum;}

    // FUSED (1 ops): row_sum=TDIVS(...)
    if (_row < 8 && _col < 1) {
        row_sum[_row][_col] = row_sum[_row][_col] / 8.0f;
    }

    // FUSED (2 ops): shifted=TROWEXPANDSUB(...); exp_scores=TEXP(...)
    if (_row < 8 && _col < 8) {
        shifted[_row][_col] = scaled[_row][_col] - row_sum[_row][0];
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

void sdpa_with_scale(float* Q_mem, float* K_mem, float* V_mem, float* output_mem) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    sdpa_with_scale_kernel<<<grid, block>>>(Q_mem, K_mem, V_mem, output_mem);
    cudaDeviceSynchronize();
}