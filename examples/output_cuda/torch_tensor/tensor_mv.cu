// PTO Program: tensor_mv
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_mv
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 320 bytes (0.3 KB)
//   Total capacity (w/ reuse): 320 bytes (0.3 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               8x1        f32        32   [  2,   3]           -
//   self                 8x8        f32       256   [  0,  -1]           -
//   vec                  8x1        f32        32   [  1,  -1]           -
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
__device__ float vec[8][1];
__device__ float result[8][1];

__global__ void tensor_mv_kernel(float* input_self, float* input_vec, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 0 loop overheads saved

    // FUSED (1 ops): self=TLOAD(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input_self[_row * 8 + _col];
    }

    // FUSED (1 ops): vec=TLOAD(...)
    if (_row < 8 && _col < 1) {
        vec[_row][_col] = input_vec[_row * 1 + _col];
    }

    // TMATMUL: result = self @ vec
    if (_row < 8 && _col < 1) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 8; _k++) _sum += self[_row][_k] * vec[_k][_col];
        result[_row][_col] = _sum;}

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 8 && _col < 1) {
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void tensor_mv(float* input_self, float* input_vec, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_mv_kernel<<<grid, block>>>(input_self, input_vec, output);
    cudaDeviceSynchronize();
}