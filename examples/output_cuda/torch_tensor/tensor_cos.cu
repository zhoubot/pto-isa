// PTO Program: tensor_cos
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_cos
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
//   ones                 8x8        f32       256   [  5,   6]           <- x4
//   result               8x8        f32       256   [  7,   8]           <- term1
//   temp                 8x8        f32       256   [  6,   7]           -
//   term1                8x8        f32       256   [  3,   6]           -
//   term2                8x8        f32       256   [  4,   7]           <- x2
//   x                    8x8        f32       256   [  0,   1]           -
//   x2                   8x8        f32       256   [  1,   3]           -
//   x4                   8x8        f32       256   [  2,   4]           <- x
//
// BUFFER REUSE MAP:
//   x4 reuses buffer of x
//   term2 reuses buffer of x2
//   ones reuses buffer of x4
//   result reuses buffer of term1
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
__device__ float x4[8][8];
__device__ float term1[8][8];
__device__ float term2[8][8];
__device__ float ones[8][8];
__device__ float temp[8][8];
__device__ float result[8][8];

__global__ void tensor_cos_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 8 loop overheads saved

    // FUSED (9 ops): x=TLOAD(...); x2=TMUL(...); x4=TMUL(...); term1=TDIVS(...); term2=TDIVS(...); ones=TEXPANDS(...); temp=TSUB(...); result=TADD(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        x2[_row][_col] = x[_row][_col] * x[_row][_col];
        x4[_row][_col] = x2[_row][_col] * x2[_row][_col];
        term1[_row][_col] = x2[_row][_col] / 2.0f;
        term2[_row][_col] = x4[_row][_col] / 24.0f;
        ones[_row][_col] = 1.0f;
        temp[_row][_col] = ones[_row][_col] - term1[_row][_col];
        result[_row][_col] = temp[_row][_col] + term2[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_cos(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_cos_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}