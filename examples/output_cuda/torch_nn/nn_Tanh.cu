// PTO Program: nn_Tanh
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_Tanh
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 1,792 bytes (1.8 KB)
//   Total capacity (w/ reuse): 1,024 bytes (1.0 KB)
//   Reuse savings:            768 bytes (42.9%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   denominator          8x8        f32       256   [  5,   6]           -
//   exp_neg_x            8x8        f32       256   [  3,   5]           <- x
//   exp_x                8x8        f32       256   [  1,   5]           -
//   neg_x                8x8        f32       256   [  2,   3]           -
//   numerator            8x8        f32       256   [  4,   6]           <- neg_x
//   result               8x8        f32       256   [  6,   7]           <- exp_x
//   x                    8x8        f32       256   [  0,   2]           -
//
// BUFFER REUSE MAP:
//   exp_neg_x reuses buffer of x
//   numerator reuses buffer of neg_x
//   result reuses buffer of exp_x
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
__device__ float exp_x[8][8];
__device__ float neg_x[8][8];
__device__ float exp_neg_x[8][8];
__device__ float numerator[8][8];
__device__ float denominator[8][8];
__device__ float result[8][8];

__global__ void nn_Tanh_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 7 loop overheads saved

    // FUSED (8 ops): x=TLOAD(...); exp_x=TEXP(...); neg_x=TNEG(...); exp_neg_x=TEXP(...); numerator=TSUB(...); denominator=TADD(...); result=TDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        exp_x[_row][_col] = __expf(x[_row][_col]);
        neg_x[_row][_col] = -x[_row][_col];
        exp_neg_x[_row][_col] = __expf(neg_x[_row][_col]);
        numerator[_row][_col] = exp_x[_row][_col] - exp_neg_x[_row][_col];
        denominator[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
        result[_row][_col] = numerator[_row][_col] / denominator[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_Tanh(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_Tanh_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}