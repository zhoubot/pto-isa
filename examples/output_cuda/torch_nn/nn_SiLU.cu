// PTO Program: nn_SiLU
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_SiLU
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
//   exp_neg              8x8        f32       256   [  2,   3]           -
//   neg_x                8x8        f32       256   [  1,   2]           -
//   one_plus             8x8        f32       256   [  3,   4]           <- neg_x
//   result               8x8        f32       256   [  5,   6]           <- one_plus
//   sigmoid_out          8x8        f32       256   [  4,   5]           <- exp_neg
//   x                    8x8        f32       256   [  0,   5]           -
//
// BUFFER REUSE MAP:
//   one_plus reuses buffer of neg_x
//   sigmoid_out reuses buffer of exp_neg
//   result reuses buffer of one_plus
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
__device__ float neg_x[8][8];
__device__ float exp_neg[8][8];
__device__ float one_plus[8][8];
__device__ float sigmoid_out[8][8];
__device__ float result[8][8];

__global__ void nn_SiLU_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 6 loop overheads saved

    // FUSED (7 ops): x=TLOAD(...); neg_x=TNEG(...); exp_neg=TEXP(...); one_plus=TADDS(...); sigmoid_out=TRECIP(...); result=TMUL(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        neg_x[_row][_col] = -x[_row][_col];
        exp_neg[_row][_col] = __expf(neg_x[_row][_col]);
        one_plus[_row][_col] = exp_neg[_row][_col] + 1.0f;
        sigmoid_out[_row][_col] = 1.0f / one_plus[_row][_col];
        result[_row][_col] = x[_row][_col] * sigmoid_out[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_SiLU(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_SiLU_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}