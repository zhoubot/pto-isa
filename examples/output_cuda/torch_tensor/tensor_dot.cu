// PTO Program: tensor_dot
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_dot
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 772 bytes (0.8 KB)
//   Total capacity (w/ reuse): 772 bytes (0.8 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   other                1x64       f32       256   [  1,   2]           -
//   prod                 1x64       f32       256   [  2,   3]           -
//   result               1x1        f32         4   [  3,   4]           -
//   self                 1x64       f32       256   [  0,   2]           -
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

__device__ float self[1][64];
__device__ float other[1][64];
__device__ float prod[1][64];
__device__ float result[1][1];

__global__ void tensor_dot_kernel(float* input_self, float* input_other, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (3 ops): self=TLOAD(...); other=TLOAD(...); prod=TMUL(...)
    if (_row < 1 && _col < 64) {
        self[_row][_col] = input_self[_row * 64 + _col];
        other[_row][_col] = input_other[_row * 64 + _col];
        prod[_row][_col] = self[_row][_col] * other[_row][_col];
    }

    // TROWSUM: result = rowsum(prod)
    if (_col == 0 && _row < 1) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 64; _c++) _sum += prod[_row][_c];
        result[_row][0] = _sum;}

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void tensor_dot(float* input_self, float* input_other, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_dot_kernel<<<grid, block>>>(input_self, input_other, output);
    cudaDeviceSynchronize();
}