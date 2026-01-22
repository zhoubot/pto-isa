// PTO Program: tensor_hypot
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_hypot
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 1,536 bytes (1.5 KB)
//   Total capacity (w/ reuse): 768 bytes (0.8 KB)
//   Reuse savings:            768 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   other                8x8        f32       256   [  1,   3]           -
//   other_sq             8x8        f32       256   [  3,   4]           <- self
//   result               8x8        f32       256   [  5,   6]           <- self_sq
//   self                 8x8        f32       256   [  0,   2]           -
//   self_sq              8x8        f32       256   [  2,   4]           -
//   sum_sq               8x8        f32       256   [  4,   5]           <- other
//
// BUFFER REUSE MAP:
//   other_sq reuses buffer of self
//   sum_sq reuses buffer of other
//   result reuses buffer of self_sq
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
__device__ float self_sq[8][8];
__device__ float other_sq[8][8];
__device__ float sum_sq[8][8];
__device__ float result[8][8];

__global__ void tensor_hypot_kernel(float* input_self, float* input_other, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 6 loop overheads saved

    // FUSED (7 ops): self=TLOAD(...); other=TLOAD(...); self_sq=TMUL(...); other_sq=TMUL(...); sum_sq=TADD(...); result=TSQRT(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        self[_row][_col] = input_self[_row * 8 + _col];
        other[_row][_col] = input_other[_row * 8 + _col];
        self_sq[_row][_col] = self[_row][_col] * self;
        other_sq[_row][_col] = other[_row][_col] * other[_row][_col];
        sum_sq[_row][_col] = self_sq[_row][_col] + other_sq[_row][_col];
        result[_row][_col] = __fsqrt_rn(sum_sq[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_hypot(float* input_self, float* input_other, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_hypot_kernel<<<grid, block>>>(input_self, input_other, output);
    cudaDeviceSynchronize();
}