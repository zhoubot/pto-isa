// PTO Program: tensor_tanh
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_tanh
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
//   denominator          8x8        f32       256   [  4,   5]           -
//   exp_2x               8x8        f32       256   [  2,   4]           <- x
//   numerator            8x8        f32       256   [  3,   5]           <- x2
//   result               8x8        f32       256   [  5,   6]           <- exp_2x
//   x                    8x8        f32       256   [  0,   1]           -
//   x2                   8x8        f32       256   [  1,   2]           -
//
// BUFFER REUSE MAP:
//   exp_2x reuses buffer of x
//   numerator reuses buffer of x2
//   result reuses buffer of exp_2x
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

__device__ float x[8][8];
__device__ float x2[8][8];
__device__ float exp_2x[8][8];
__device__ float numerator[8][8];
__device__ float denominator[8][8];
__device__ float result[8][8];

__global__ void tensor_tanh_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 6 loop overheads saved

    // FUSED (7 ops): x=TLOAD(...); x2=TMULS(...); exp_2x=TEXP(...); numerator=TADDS(...); denominator=TADDS(...); result=TDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        x2[_row][_col] = x[_row][_col] * 2.0f;
        exp_2x[_row][_col] = __expf(x2[_row][_col]);
        numerator[_row][_col] = exp_2x[_row][_col] + -1.0f;
        denominator[_row][_col] = exp_2x[_row][_col] + 1.0f;
        result[_row][_col] = numerator[_row][_col] / denominator[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_tanh(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_tanh_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}