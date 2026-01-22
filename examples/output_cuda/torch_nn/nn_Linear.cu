// PTO Program: nn_Linear
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_Linear
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
//   bias                 8x8        f32       256   [  2,   4]           -
//   mm_result            8x8        f32       256   [  3,   4]           -
//   result               8x8        f32       256   [  4,   5]           -
//   weight               8x8        f32       256   [  1,  -1]           -
//   x                    8x8        f32       256   [  0,  -1]           -
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
__device__ float weight[8][8];
__device__ float bias[8][8];
__device__ float mm_result[8][8];
__device__ float result[8][8];

__global__ void nn_Linear_kernel(float* input, float* weight_mem, float* bias_mem, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (3 ops): x=TLOAD(...); weight=TLOAD(...); bias=TLOAD(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        weight[_row][_col] = weight_mem[_row * 8 + _col];
        bias[_row][_col] = bias_mem[_row * 8 + _col];
    }

    // TMATMUL: mm_result = x @ weight
    if (_row < 8 && _col < 8) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 8; _k++) _sum += x[_row][_k] * weight[_k][_col];
        mm_result[_row][_col] = _sum;}

    // FUSED (2 ops): result=TADD(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        result[_row][_col] = mm_result[_row][_col] + bias[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_Linear(float* input, float* weight_mem, float* bias_mem, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_Linear_kernel<<<grid, block>>>(input, weight_mem, bias_mem, output);
    cudaDeviceSynchronize();
}