// PTO Program: nn_GELU
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_GELU
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 1,792 bytes (1.8 KB)
//   Total capacity (w/ reuse): 768 bytes (0.8 KB)
//   Reuse savings:            1,024 bytes (57.1%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg              8x8        f32       256   [  3,   4]           <- scaled_x
//   neg_scaled           8x8        f32       256   [  2,   3]           -
//   one_plus             8x8        f32       256   [  4,   5]           <- neg_scaled
//   result               8x8        f32       256   [  6,   7]           <- one_plus
//   scaled_x             8x8        f32       256   [  1,   2]           -
//   sigmoid_out          8x8        f32       256   [  5,   6]           <- exp_neg
//   x                    8x8        f32       256   [  0,   6]           -
//
// BUFFER REUSE MAP:
//   exp_neg reuses buffer of scaled_x
//   one_plus reuses buffer of neg_scaled
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
__device__ float scaled_x[8][8];
__device__ float neg_scaled[8][8];
__device__ float exp_neg[8][8];
__device__ float one_plus[8][8];
__device__ float sigmoid_out[8][8];
__device__ float result[8][8];

__global__ void nn_GELU_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 7 loop overheads saved

    // FUSED (8 ops): x=TLOAD(...); scaled_x=TMULS(...); neg_scaled=TNEG(...); exp_neg=TEXP(...); one_plus=TADDS(...); sigmoid_out=TRECIP(...); result=TMUL(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        scaled_x[_row][_col] = x[_row][_col] * 1.702f;
        neg_scaled[_row][_col] = -scaled_x[_row][_col];
        exp_neg[_row][_col] = __expf(neg_scaled[_row][_col]);
        one_plus[_row][_col] = exp_neg[_row][_col] + 1.0f;
        sigmoid_out[_row][_col] = 1.0f / one_plus[_row][_col];
        result[_row][_col] = x[_row][_col] * sigmoid_out[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_GELU(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_GELU_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}