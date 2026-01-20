// PTO Program: tile_rowexpandmul_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_rowexpandmul_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 65,792 bytes (64.2 KB)
//   Total capacity (w/ reuse): 65,792 bytes (64.2 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               64x128     f32     32768   [  2,   3]           -
//   row_vals             64x1       f32       256   [  1,   2]           -
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
__device__ float row_vals[64][1];
__device__ float result[64][128];

__global__ void tile_rowexpandmul_64_kernel(float* input_x, float* input_row, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 1 loop overheads saved

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 64 && _col < 128) {
        x[_row][_col] = input_x[_row * 128 + _col];
    }

    // FUSED (1 ops): row_vals=TLOAD(...)
    if (_row < 64 && _col < 1) {
        row_vals[_row][_col] = input_row[_row * 1 + _col];
    }

    // FUSED (2 ops): result=TROWEXPANDMUL(...); output=TSTORE(...)
    if (_row < 64 && _col < 128) {
        result[_row][_col] = x[_row][_col] * row_vals[_row][0];
        output[_row * 128 + _col] = result[_row][_col];
    }

}

void tile_rowexpandmul_64(float* input_x, float* input_row, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tile_rowexpandmul_64_kernel<<<grid, block>>>(input_x, input_row, output);
    cudaDeviceSynchronize();
}