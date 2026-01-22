// PTO Program: nn_Hardswish
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_Hardswish
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
//   relu6_out            8x8        f32       256   [  4,   5]           -
//   relu_out             8x8        f32       256   [  2,   4]           -
//   result               8x8        f32       256   [  6,   7]           <- x
//   scaled               8x8        f32       256   [  5,   6]           <- relu_out
//   six                  8x8        f32       256   [  3,   4]           <- x_plus_3
//   x                    8x8        f32       256   [  0,   5]           -
//   x_plus_3             8x8        f32       256   [  1,   2]           -
//
// BUFFER REUSE MAP:
//   six reuses buffer of x_plus_3
//   scaled reuses buffer of relu_out
//   result reuses buffer of x
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
__device__ float x_plus_3[8][8];
__device__ float relu_out[8][8];
__device__ float six[8][8];
__device__ float relu6_out[8][8];
__device__ float scaled[8][8];
__device__ float result[8][8];

__global__ void nn_Hardswish_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 7 loop overheads saved

    // FUSED (8 ops): x=TLOAD(...); x_plus_3=TADDS(...); relu_out=TRELU(...); six=TEXPANDS(...); relu6_out=TMIN(...); scaled=TMUL(...); result=TDIVS(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        x_plus_3[_row][_col] = x[_row][_col] + 3.0f;
        relu_out[_row][_col] = fmaxf(x_plus_3[_row][_col], 0.0f);
        six[_row][_col] = 6.0f;
        relu6_out[_row][_col] = fminf(relu_out[_row][_col], six[_row][_col]);
        scaled[_row][_col] = x[_row][_col] * relu6_out[_row][_col];
        result[_row][_col] = scaled[_row][_col] / 6.0f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_Hardswish(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_Hardswish_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}