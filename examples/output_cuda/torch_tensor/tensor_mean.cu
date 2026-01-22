// PTO Program: tensor_mean
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_mean
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 296 bytes (0.3 KB)
//   Total capacity (w/ reuse): 296 bytes (0.3 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               1x1        f32         4   [  3,   4]           -
//   row_sum              8x1        f32        32   [  1,   2]           -
//   self                 8x8        f32       256   [  0,   1]           -
//   total                1x1        f32         4   [  2,   3]           -
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
__device__ float row_sum[8][1];
__device__ float total[1][1];
__device__ float result[1][1];

__global__ void tensor_mean_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 1 loop overheads saved

    // FUSED (1 ops): self=TLOAD(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input[_row * 8 + _col];
    }

    // TROWSUM: row_sum = rowsum(self)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += self[_row][_c];
        row_sum[_row][0] = _sum;}

    // TCOLSUM: Not implemented

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = total[_row][_col] / 64.0f;
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void tensor_mean(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_mean_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}