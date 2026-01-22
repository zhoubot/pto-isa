// PTO Program: F_nll_loss
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_nll_loss
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 804 bytes (0.8 KB)
//   Total capacity (w/ reuse): 804 bytes (0.8 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   log_probs            8x8        f32       256   [  0,   2]           -
//   result               1x1        f32         4   [  4,   7]           -
//   row_sum              8x1        f32        32   [  3,   4]           -
//   target               8x8        f32       256   [  1,   2]           -
//   weighted             8x8        f32       256   [  2,   3]           -
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

__device__ float log_probs[8][8];
__device__ float target[8][8];
__device__ float weighted[8][8];
__device__ float row_sum[8][1];
__device__ float result[1][1];

__global__ void F_nll_loss_kernel(float* input, float* target_mem, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (3 ops): log_probs=TLOAD(...); target=TLOAD(...); weighted=TMUL(...)
    if (_row < 8 && _col < 8) {
        log_probs[_row][_col] = input[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
        weighted[_row][_col] = target[_row][_col] * log_probs[_row][_col];
    }

    // TROWSUM: row_sum = rowsum(weighted)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += weighted[_row][_c];
        row_sum[_row][0] = _sum;}

    // TCOLSUM: Not implemented

    // FUSED (3 ops): result=TNEG(...); result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = -result[_row][_col];
        result[_row][_col] = result[_row][_col] / 8.0f;
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void F_nll_loss(float* input, float* target_mem, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_nll_loss_kernel<<<grid, block>>>(input, target_mem, output);
    cudaDeviceSynchronize();
}