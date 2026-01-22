// PTO Program: F_threshold
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_threshold
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 1,024 bytes (1.0 KB)
//   Total capacity (w/ reuse): 1,024 bytes (1.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               8x8        f32       256   [  3,   4]           -
//   thresh_tile          8x8        f32       256   [  1,   3]           -
//   value_tile           8x8        f32       256   [  2,  -1]           -
//   x                    8x8        f32       256   [  0,   3]           -
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
__device__ float thresh_tile[8][8];
__device__ float value_tile[8][8];
__device__ float result[8][8];

__global__ void F_threshold_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (5 ops): x=TLOAD(...); thresh_tile=TEXPANDS(...); value_tile=TEXPANDS(...); result=TMAX(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        thresh_tile[_row][_col] = 0.0f;
        value_tile[_row][_col] = 0.0f;
        result[_row][_col] = fmaxf(x[_row][_col], thresh_tile[_row][_col]);
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_threshold(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_threshold_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}