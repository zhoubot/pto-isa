// PTO Program: tensor_mm
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_mm
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
//   mat2                 8x8        f32       256   [  1,  -1]           -
//   result               8x8        f32       256   [  2,   3]           -
//   self                 8x8        f32       256   [  0,  -1]           -
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
__device__ float mat2[8][8];
__device__ float result[8][8];

__global__ void tensor_mm_kernel(float* input_self, float* input_mat2, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 1 loop overheads saved

    // FUSED (2 ops): self=TLOAD(...); mat2=TLOAD(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input_self[_row * 8 + _col];
        mat2[_row][_col] = input_mat2[_row * 8 + _col];
    }

    // TMATMUL: result = self @ mat2
    if (_row < 8 && _col < 8) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 8; _k++) _sum += self[_row][_k] * mat2[_k][_col];
        result[_row][_col] = _sum;}

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_mm(float* input_self, float* input_mat2, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_mm_kernel<<<grid, block>>>(input_self, input_mat2, output);
    cudaDeviceSynchronize();
}