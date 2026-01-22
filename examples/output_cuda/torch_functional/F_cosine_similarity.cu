// PTO Program: F_cosine_similarity
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_cosine_similarity
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     12
//   Total capacity (no reuse): 1,504 bytes (1.5 KB)
//   Total capacity (w/ reuse): 896 bytes (0.9 KB)
//   Reuse savings:            608 bytes (40.4%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   dot_prod             8x8        f32       256   [  2,   3]           -
//   dot_sum              8x1        f32        32   [  3,  12]           -
//   norm_prod            8x1        f32        32   [ 10,  12]           <- x2_norm_sq
//   result               8x1        f32        32   [ 12,  13]           <- x1_norm
//   x1                   8x8        f32       256   [  0,   4]           -
//   x1_norm              8x1        f32        32   [  8,  10]           -
//   x1_norm_sq           8x1        f32        32   [  6,   8]           -
//   x1_sq                8x8        f32       256   [  4,   6]           <- dot_prod
//   x2                   8x8        f32       256   [  1,   5]           -
//   x2_norm              8x1        f32        32   [  9,  10]           <- x1_norm_sq
//   x2_norm_sq           8x1        f32        32   [  7,   9]           -
//   x2_sq                8x8        f32       256   [  5,   7]           <- x1
//
// BUFFER REUSE MAP:
//   x1_sq reuses buffer of dot_prod
//   x2_sq reuses buffer of x1
//   x2_norm reuses buffer of x1_norm_sq
//   norm_prod reuses buffer of x2_norm_sq
//   result reuses buffer of x1_norm
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

__device__ float x1[8][8];
__device__ float x2[8][8];
__device__ float dot_prod[8][8];
__device__ float x1_sq[8][8];
__device__ float x2_sq[8][8];
__device__ float dot_sum[8][1];
__device__ float x1_norm_sq[8][1];
__device__ float x2_norm_sq[8][1];
__device__ float x1_norm[8][1];
__device__ float x2_norm[8][1];
__device__ float norm_prod[8][1];
__device__ float result[8][1];

__global__ void F_cosine_similarity_kernel(float* input1, float* input2, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 8 loop overheads saved

    // FUSED (3 ops): x1=TLOAD(...); x2=TLOAD(...); dot_prod=TMUL(...)
    if (_row < 8 && _col < 8) {
        x1[_row][_col] = input1[_row * 8 + _col];
        x2[_row][_col] = input2[_row * 8 + _col];
        dot_prod[_row][_col] = x1[_row][_col] * x2[_row][_col];
    }

    // TROWSUM: dot_sum = rowsum(dot_prod)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += dot_prod[_row][_c];
        dot_sum[_row][0] = _sum;}

    // FUSED (2 ops): x1_sq=TMUL(...); x2_sq=TMUL(...)
    if (_row < 8 && _col < 8) {
        x1_sq[_row][_col] = x1[_row][_col] * x1[_row][_col];
        x2_sq[_row][_col] = x2[_row][_col] * x2[_row][_col];
    }

    // TROWSUM: x1_norm_sq = rowsum(x1_sq)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += x1_sq[_row][_c];
        x1_norm_sq[_row][0] = _sum;}

    // TROWSUM: x2_norm_sq = rowsum(x2_sq)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += x2_sq[_row][_c];
        x2_norm_sq[_row][0] = _sum;}

    // FUSED (6 ops): x1_norm=TSQRT(...); x2_norm=TSQRT(...); norm_prod=TMUL(...); norm_prod=TADDS(...); result=TDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 1) {
        x1_norm[_row][_col] = __fsqrt_rn(x1_norm_sq[_row][_col]);
        x2_norm[_row][_col] = __fsqrt_rn(x2_norm_sq[_row][_col]);
        norm_prod[_row][_col] = x1_norm[_row][_col] * x2_norm[_row][_col];
        norm_prod[_row][_col] = norm_prod[_row][_col] + 1e-08f;
        result[_row][_col] = dot_sum[_row][_col] / norm_prod[_row][_col];
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void F_cosine_similarity(float* input1, float* input2, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_cosine_similarity_kernel<<<grid, block>>>(input1, input2, output);
    cudaDeviceSynchronize();
}