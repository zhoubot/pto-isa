// PTO Program: F_bilinear
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_bilinear
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 1,280 bytes (1.2 KB)
//   Total capacity (w/ reuse): 1,280 bytes (1.2 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   output               8x8        f32       256   [  4,   5]           -
//   temp                 8x8        f32       256   [  3,   4]           -
//   weight               8x8        f32       256   [  2,  -1]           -
//   x1                   8x8        f32       256   [  0,  -1]           -
//   x2                   8x8        f32       256   [  1,   4]           -
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
__device__ float weight[8][8];
__device__ float temp[8][8];
__device__ float output[8][8];

__global__ void F_bilinear_kernel(float* input1, float* input2, float* weight_mem, float* output_mem) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (3 ops): x1=TLOAD(...); x2=TLOAD(...); weight=TLOAD(...)
    if (_row < 8 && _col < 8) {
        x1[_row][_col] = input1[_row * 8 + _col];
        x2[_row][_col] = input2[_row * 8 + _col];
        weight[_row][_col] = weight_mem[_row * 8 + _col];
    }

    // TMATMUL: temp = x1 @ weight
    if (_row < 8 && _col < 8) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 8; _k++) _sum += x1[_row][_k] * weight[_k][_col];
        temp[_row][_col] = _sum;}

    // FUSED (2 ops): output=TMUL(...); output_mem=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output[_row][_col] = temp[_row][_col] * x2[_row][_col];
        output_mem[_row * 8 + _col] = output[_row][_col];
    }

}

void F_bilinear(float* input1, float* input2, float* weight_mem, float* output_mem) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_bilinear_kernel<<<grid, block>>>(input1, input2, weight_mem, output_mem);
    cudaDeviceSynchronize();
}