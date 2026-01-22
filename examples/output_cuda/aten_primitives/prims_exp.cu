// PTO Program: prims_exp
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: prims_exp
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 32,768 bytes (32.0 KB)
//   Total capacity (w/ reuse): 32,768 bytes (32.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               1x4096     f32     16384   [  4,  11]           -
//   x                    1x4096     f32     16384   [  3,  10]           -
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

__device__ float x[1][4096];
__device__ float result[1][4096];

__global__ void prims_exp_kernel(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED (3 ops): x=TLOAD(...); result=TEXP(...); output=TSTORE(...)
        if (_row < 1 && _col < 4096) {
            x[_row][_col] = input[(tile_idx) * 4096 + _row * 4096 + _col];
            result[_row][_col] = __expf(x[_row][_col]);
            output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED (3 ops): x=TLOAD(...); result=TEXP(...); output=TSTORE(...)
        if (_row < 1 && _col < 4096) {
            x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 4096 + _col];
            result[_row][_col] = __expf(x[_row][_col]);
            output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
        }

    }

}

void prims_exp(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    prims_exp_kernel<<<grid, block>>>(input, output, num_full_tiles, tail_elements, zero, tile_size);
    cudaDeviceSynchronize();
}