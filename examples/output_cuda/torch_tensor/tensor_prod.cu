// PTO Program: tensor_prod
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_prod
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 552 bytes (0.5 KB)
//   Total capacity (w/ reuse): 552 bytes (0.5 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   log_self             8x8        f32       256   [  1,   2]           -
//   result               1x1        f32         4   [  4,   5]           -
//   row_sum              8x1        f32        32   [  2,   3]           -
//   self                 8x8        f32       256   [  0,   1]           -
//   total                1x1        f32         4   [  3,   4]           -
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
__device__ float log_self[8][8];
__device__ float row_sum[8][1];
__device__ float total[1][1];
__device__ float result[1][1];

__global__ void tensor_prod_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (2 ops): self=TLOAD(...); log_self=TLOG(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input[_row * 8 + _col];
        log_self[_row][_col] = __logf(self[_row][_col]);
    }

    // TROWSUM: row_sum = rowsum(log_self)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += log_self[_row][_c];
        row_sum[_row][0] = _sum;}

    // TCOLSUM: Not implemented

    // FUSED (2 ops): result=TEXP(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = __expf(total[_row][_col]);
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void tensor_prod(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_prod_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}