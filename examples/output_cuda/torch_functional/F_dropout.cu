// PTO Program: F_dropout
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_dropout
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 512 bytes (0.5 KB)
//   Total capacity (w/ reuse): 512 bytes (0.5 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               8x8        f32       256   [  1,   2]           -
//   x                    8x8        f32       256   [  0,   1]           -
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
__device__ float result[8][8];

__global__ void F_dropout_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (3 ops): x=TLOAD(...); result=TMULS(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        result[_row][_col] = x[_row][_col] * 1.0f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_dropout(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_dropout_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}