// PTO Program: tensor_log_softmax
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_log_softmax
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 1,120 bytes (1.1 KB)
//   Total capacity (w/ reuse): 576 bytes (0.6 KB)
//   Reuse savings:            544 bytes (48.6%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_shifted          8x8        f32       256   [  4,   5]           <- self
//   log_sum              8x1        f32        32   [  6,   7]           -
//   result               8x8        f32       256   [  7,   8]           <- exp_shifted
//   row_mean             8x1        f32        32   [  1,   3]           -
//   row_sum              8x1        f32        32   [  5,   6]           <- row_mean
//   self                 8x8        f32       256   [  0,   3]           -
//   shifted              8x8        f32       256   [  3,   7]           -
//
// BUFFER REUSE MAP:
//   exp_shifted reuses buffer of self
//   row_sum reuses buffer of row_mean
//   result reuses buffer of exp_shifted
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

__device__ float self[8][8];
__device__ float row_mean[8][1];
__device__ float shifted[8][8];
__device__ float exp_shifted[8][8];
__device__ float row_sum[8][1];
__device__ float log_sum[8][1];
__device__ float result[8][8];

__global__ void tensor_log_softmax_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (1 ops): self=TLOAD(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input[_row * 8 + _col];
    }

    // TROWSUM: row_mean = rowsum(self)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += self[_row][_c];
        row_mean[_row][0] = _sum;}

    // FUSED (1 ops): row_mean=TDIVS(...)
    if (_row < 8 && _col < 1) {
        row_mean[_row][_col] = row_mean[_row][_col] / 8.0f;
    }

    // FUSED (2 ops): shifted=TROWEXPANDSUB(...); exp_shifted=TEXP(...)
    if (_row < 8 && _col < 8) {
        shifted[_row][_col] = self[_row][_col] - row_mean[_row][0];
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

    // FUSED (2 ops): result=TROWEXPANDSUB(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        result[_row][_col] = shifted[_row][_col] - log_sum[_row][0];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_log_softmax(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_log_softmax_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}