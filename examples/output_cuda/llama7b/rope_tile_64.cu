// PTO Program: rope_tile_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rope_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 196,608 bytes (192.0 KB)
//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)
//   Reuse savings:            65,536 bytes (33.3%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   cos_pos              64x128     f32     32768   [  1,   3]           -
//   result               64x128     f32     32768   [  5,   6]           <- x
//   sin_pos              64x128     f32     32768   [  2,   4]           -
//   x                    64x128     f32     32768   [  0,   4]           -
//   x_cos                64x128     f32     32768   [  3,   5]           -
//   x_sin                64x128     f32     32768   [  4,   5]           <- cos_pos
//
// BUFFER REUSE MAP:
//   x_sin reuses buffer of cos_pos
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

__device__ float x[64][128];
__device__ float cos_pos[64][128];
__device__ float sin_pos[64][128];
__device__ float x_cos[64][128];
__device__ float x_sin[64][128];
__device__ float result[64][128];

__global__ void rope_tile_64_kernel(float* input, float* cos_cache, float* sin_cache, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 6 loop overheads saved

    // FUSED (7 ops): x=TLOAD(...); cos_pos=TLOAD(...); sin_pos=TLOAD(...); x_cos=TMUL(...); x_sin=TMUL(...); result=TADD(...); output=TSTORE(...)
    if (_row < 64 && _col < 128) {
        x[_row][_col] = input[_row * 128 + _col];
        cos_pos[_row][_col] = cos_cache[_row * 128 + _col];
        sin_pos[_row][_col] = sin_cache[_row * 128 + _col];
        x_cos[_row][_col] = x[_row][_col] * cos_pos[_row][_col];
        x_sin[_row][_col] = x[_row][_col] * sin_pos[_row][_col];
        result[_row][_col] = x_cos[_row][_col] + x_sin[_row][_col];
        output[_row * 128 + _col] = result[_row][_col];
    }

}

void rope_tile_64(float* input, float* cos_cache, float* sin_cache, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    rope_tile_64_kernel<<<grid, block>>>(input, cos_cache, sin_cache, output);
    cudaDeviceSynchronize();
}