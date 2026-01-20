// PTO Program: tile_add_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_add_64
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
//   a                    64x128     f32     32768   [  0,   2]           -
//   b                    64x128     f32     32768   [  1,   2]           -
//   result               64x128     f32     32768   [  2,   3]           -
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

__device__ float a[64][128];
__device__ float b[64][128];
__device__ float result[64][128];

__global__ void tile_add_64_kernel(float* input_a, float* input_b, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (4 ops): a=TLOAD(...); b=TLOAD(...); result=TADD(...); output=TSTORE(...)
    if (_row < 64 && _col < 128) {
        a[_row][_col] = input_a[_row * 128 + _col];
        b[_row][_col] = input_b[_row * 128 + _col];
        result[_row][_col] = a[_row][_col] + b[_row][_col];
        output[_row * 128 + _col] = result[_row][_col];
    }

}

void tile_add_64(float* input_a, float* input_b, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tile_add_64_kernel<<<grid, block>>>(input_a, input_b, output);
    cudaDeviceSynchronize();
}