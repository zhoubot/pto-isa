// PTO Program: sinh_taylor
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: sinh_taylor
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 65,536 bytes (64.0 KB)
//   Total capacity (w/ reuse): 65,536 bytes (64.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               32x128     f32     16384   [  4,  51]           -
//   term                 32x128     f32     16384   [  6,  50]           -
//   x                    32x128     f32     16384   [  3,  32]           -
//   x_squared            32x128     f32     16384   [  5,  48]           -
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

__device__ float x[32][128];
__device__ float x_squared[32][128];
__device__ float term[32][128];
__device__ float result[32][128];

__global__ void sinh_taylor_kernel(float* input, float* output, int32_t total_elements, int32_t num_full_tiles, int32_t tail_elements, int32_t offset) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 44 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED (23 ops): x=TLOAD(...); result=TMULS(...); x_squared=TMUL(...); term=TMULS(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); output=TSTORE(...)
        if (_row < 32 && _col < 128) {
            x[_row][_col] = input[(tile_idx) * 4096 + _row * 128 + _col];
            result[_row][_col] = x[_row][_col] * 1.0f;
            x_squared[_row][_col] = x[_row][_col] * x[_row][_col];
            term[_row][_col] = x[_row][_col] * 1.0f;
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 6.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 20.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 42.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 72.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 110.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 156.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            output[(tile_idx) * 4096 + _row * 128 + _col] = result[_row][_col];
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED (23 ops): x=TLOAD(...); result=TMULS(...); x_squared=TMUL(...); term=TMULS(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); output=TSTORE(...)
        if (_row < 32 && _col < 128) {
            x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 128 + _col];
            result[_row][_col] = x[_row][_col] * 1.0f;
            x_squared[_row][_col] = x[_row][_col] * x[_row][_col];
            term[_row][_col] = x[_row][_col] * 1.0f;
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 6.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 20.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 42.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 72.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 110.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 156.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            output[(num_full_tiles) * 4096 + _row * 128 + _col] = result[_row][_col];
        }

    }

}

void sinh_taylor(float* input, float* output, int32_t total_elements, int32_t num_full_tiles, int32_t tail_elements, int32_t offset) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    sinh_taylor_kernel<<<grid, block>>>(input, output, total_elements, tile_size, num_full_tiles, tail_elements, offset, zero);
    cudaDeviceSynchronize();
}