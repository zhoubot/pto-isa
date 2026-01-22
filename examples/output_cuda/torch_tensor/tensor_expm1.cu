// PTO Program: tensor_expm1
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_expm1
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 768 bytes (0.8 KB)
//   Total capacity (w/ reuse): 512 bytes (0.5 KB)
//   Reuse savings:            256 bytes (33.3%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_val              8x8        f32       256   [  1,   2]           -
//   result               8x8        f32       256   [  2,   3]           <- self
//   self                 8x8        f32       256   [  0,   1]           -
//
// BUFFER REUSE MAP:
//   result reuses buffer of self
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
__device__ float exp_val[8][8];
__device__ float result[8][8];

__global__ void tensor_expm1_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (4 ops): self=TLOAD(...); exp_val=TEXP(...); result=TADDS(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input[_row * 8 + _col];
        exp_val[_row][_col] = __expf(self[_row][_col]);
        result[_row][_col] = exp_val[_row][_col] + -1.0f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_expm1(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_expm1_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}