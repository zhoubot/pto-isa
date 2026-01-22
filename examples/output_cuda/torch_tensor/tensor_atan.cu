// PTO Program: tensor_atan
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_atan
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 2,048 bytes (2.0 KB)
//   Total capacity (w/ reuse): 1,024 bytes (1.0 KB)
//   Reuse savings:            1,024 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               8x8        f32       256   [  7,   8]           <- x
//   temp                 8x8        f32       256   [  6,   7]           <- x5
//   term1                8x8        f32       256   [  4,   6]           <- x2
//   term2                8x8        f32       256   [  5,   7]           <- x3
//   x                    8x8        f32       256   [  0,   6]           -
//   x2                   8x8        f32       256   [  1,   3]           -
//   x3                   8x8        f32       256   [  2,   4]           -
//   x5                   8x8        f32       256   [  3,   5]           -
//
// BUFFER REUSE MAP:
//   term1 reuses buffer of x2
//   term2 reuses buffer of x3
//   temp reuses buffer of x5
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
__device__ float x2[8][8];
__device__ float x3[8][8];
__device__ float x5[8][8];
__device__ float term1[8][8];
__device__ float term2[8][8];
__device__ float temp[8][8];
__device__ float result[8][8];

__global__ void tensor_atan_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 8 loop overheads saved

    // FUSED (9 ops): x=TLOAD(...); x2=TMUL(...); x3=TMUL(...); x5=TMUL(...); term1=TDIVS(...); term2=TDIVS(...); temp=TSUB(...); result=TADD(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        x2[_row][_col] = x[_row][_col] * x[_row][_col];
        x3[_row][_col] = x2[_row][_col] * x[_row][_col];
        x5[_row][_col] = x3[_row][_col] * x2[_row][_col];
        term1[_row][_col] = x3[_row][_col] / 3.0f;
        term2[_row][_col] = x5[_row][_col] / 5.0f;
        temp[_row][_col] = x[_row][_col] - term1[_row][_col];
        result[_row][_col] = temp[_row][_col] + term2[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_atan(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_atan_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}