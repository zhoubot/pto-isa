// PTO Program: tensor_max_elementwise
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_max_elementwise
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 768 bytes (0.8 KB)
//   Total capacity (w/ reuse): 768 bytes (0.8 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   other                8x8        f32       256   [  1,   2]           -
//   result               8x8        f32       256   [  2,   3]           -
//   self                 8x8        f32       256   [  0,   2]           -
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
__device__ float other[8][8];
__device__ float result[8][8];

__global__ void tensor_max_elementwise_kernel(float* input_self, float* input_other, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (4 ops): self=TLOAD(...); other=TLOAD(...); result=TMAX(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input_self[_row * 8 + _col];
        other[_row][_col] = input_other[_row * 8 + _col];
        result[_row][_col] = fmaxf(self[_row][_col], other[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_max_elementwise(float* input_self, float* input_other, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_max_elementwise_kernel<<<grid, block>>>(input_self, input_other, output);
    cudaDeviceSynchronize();
}