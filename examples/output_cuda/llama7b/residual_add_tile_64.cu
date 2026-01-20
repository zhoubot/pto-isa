// PTO Program: residual_add_tile_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: residual_add_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 98,304 bytes (96.0 KB)
//   Total capacity (w/ reuse): 98,304 bytes (96.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   residual             64x128     f32     32768   [  1,   2]           -
//   result               64x128     f32     32768   [  2,   3]           -
//   x                    64x128     f32     32768   [  0,   2]           -
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

__device__ float x[64][128];
__device__ float residual[64][128];
__device__ float result[64][128];

__global__ void residual_add_tile_64_kernel(float* input, float* input_residual, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (4 ops): x=TLOAD(...); residual=TLOAD(...); result=TADD(...); output=TSTORE(...)
    if (_row < 64 && _col < 128) {
        x[_row][_col] = input[_row * 128 + _col];
        residual[_row][_col] = input_residual[_row * 128 + _col];
        result[_row][_col] = x[_row][_col] + residual[_row][_col];
        output[_row * 128 + _col] = result[_row][_col];
    }

}

void residual_add_tile_64(float* input, float* input_residual, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    residual_add_tile_64_kernel<<<grid, block>>>(input, input_residual, output);
    cudaDeviceSynchronize();
}