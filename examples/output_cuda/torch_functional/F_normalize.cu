// PTO Program: F_normalize
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_normalize
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 832 bytes (0.8 KB)
//   Total capacity (w/ reuse): 576 bytes (0.6 KB)
//   Reuse savings:            256 bytes (30.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   norm                 8x1        f32        32   [  3,   5]           -
//   result               8x8        f32       256   [  5,   6]           <- x_sq
//   row_sum              8x1        f32        32   [  2,   3]           -
//   x                    8x8        f32       256   [  0,   5]           -
//   x_sq                 8x8        f32       256   [  1,   2]           -
//
// BUFFER REUSE MAP:
//   result reuses buffer of x_sq
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

__device__ float x[8][8];
__device__ float x_sq[8][8];
__device__ float row_sum[8][1];
__device__ float norm[8][1];
__device__ float result[8][8];

__global__ void F_normalize_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (2 ops): x=TLOAD(...); x_sq=TMUL(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        x_sq[_row][_col] = x[_row][_col] * x[_row][_col];
    }

    // TROWSUM: row_sum = rowsum(x_sq)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += x_sq[_row][_c];
        row_sum[_row][0] = _sum;}

    // FUSED (2 ops): norm=TSQRT(...); norm=TADDS(...)
    if (_row < 8 && _col < 1) {
        norm[_row][_col] = __fsqrt_rn(row_sum[_row][_col]);
        norm[_row][_col] = norm[_row][_col] + 1e-12f;
    }

    // FUSED (2 ops): result=TROWEXPANDDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        result[_row][_col] = x[_row][_col] / norm[_row][0];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_normalize(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_normalize_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}