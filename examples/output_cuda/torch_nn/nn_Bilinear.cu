// PTO Program: nn_Bilinear
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_Bilinear
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 1,280 bytes (1.2 KB)
//   Total capacity (w/ reuse): 1,024 bytes (1.0 KB)
//   Reuse savings:            256 bytes (20.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   product              8x8        f32       256   [  3,  -1]           -
//   result               8x8        f32       256   [  4,   5]           <- x1
//   weight               8x8        f32       256   [  2,  -1]           -
//   x1                   8x8        f32       256   [  0,   3]           -
//   x2                   8x8        f32       256   [  1,   3]           -
//
// BUFFER REUSE MAP:
//   result reuses buffer of x1
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

__device__ float x1[8][8];
__device__ float x2[8][8];
__device__ float product[8][8];
__device__ float weight[8][8];
__device__ float result[8][8];

__global__ void nn_Bilinear_kernel(float* input1, float* input2, float* weight_mem, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (4 ops): x1=TLOAD(...); x2=TLOAD(...); weight=TLOAD(...); product=TMUL(...)
    if (_row < 8 && _col < 8) {
        x1[_row][_col] = input1[_row * 8 + _col];
        x2[_row][_col] = input2[_row * 8 + _col];
        weight[_row][_col] = weight_mem[_row * 8 + _col];
        product[_row][_col] = x1[_row][_col] * x2[_row][_col];
    }

    // TMATMUL: result = product @ weight
    if (_row < 8 && _col < 8) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 8; _k++) _sum += product[_row][_k] * weight[_k][_col];
        result[_row][_col] = _sum;}

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_Bilinear(float* input1, float* input2, float* weight_mem, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_Bilinear_kernel<<<grid, block>>>(input1, input2, weight_mem, output);
    cudaDeviceSynchronize();
}