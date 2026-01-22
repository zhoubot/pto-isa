// PTO Program: tensor_sign
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_sign
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 1,024 bytes (1.0 KB)
//   Total capacity (w/ reuse): 768 bytes (0.8 KB)
//   Reuse savings:            256 bytes (25.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   abs_plus_eps         8x8        f32       256   [  2,   3]           -
//   abs_self             8x8        f32       256   [  1,   2]           -
//   result               8x8        f32       256   [  3,   4]           <- abs_self
//   self                 8x8        f32       256   [  0,   3]           -
//
// BUFFER REUSE MAP:
//   result reuses buffer of abs_self
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
__device__ float abs_self[8][8];
__device__ float abs_plus_eps[8][8];
__device__ float result[8][8];

__global__ void tensor_sign_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (5 ops): self=TLOAD(...); abs_self=TABS(...); abs_plus_eps=TADDS(...); result=TDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input[_row * 8 + _col];
        abs_self[_row][_col] = fabsf(self[_row][_col]);
        abs_plus_eps[_row][_col] = abs_self[_row][_col] + 1e-07f;
        result[_row][_col] = self[_row][_col] / abs_plus_eps[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_sign(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_sign_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}