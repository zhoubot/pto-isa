// PTO Program: prims_sum_row
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: prims_sum_row
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 16,388 bytes (16.0 KB)
//   Total capacity (w/ reuse): 16,388 bytes (16.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               1x1        f32         4   [  4,  11]           -
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
__device__ float result[1][1];

__global__ void prims_sum_row_kernel(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 0 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED (1 ops): x=TLOAD(...)
        if (_row < 1 && _col < 4096) {
            x[_row][_col] = input[(tile_idx) * 4096 + _row * 4096 + _col];
        }

        // TROWSUM: result = rowsum(x)
        if (_col == 0 && _row < 1) {
            float _sum = 0.0f;
            for (int _c = 0; _c < 4096; _c++) _sum += x[_row][_c];
            result[_row][0] = _sum;}

        // FUSED (1 ops): output=TSTORE(...)
        if (_row < 1 && _col < 1) {
            output[(tile_idx) * 1 + _row * 1 + _col] = result[_row][_col];
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED (1 ops): x=TLOAD(...)
        if (_row < 1 && _col < 4096) {
            x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 4096 + _col];
        }

        // TROWSUM: result = rowsum(x)
        if (_col == 0 && _row < 1) {
            float _sum = 0.0f;
            for (int _c = 0; _c < 4096; _c++) _sum += x[_row][_c];
            result[_row][0] = _sum;}

        // FUSED (1 ops): output=TSTORE(...)
        if (_row < 1 && _col < 1) {
            output[(num_full_tiles) * 1 + _row * 1 + _col] = result[_row][_col];
        }

    }

}

void prims_sum_row(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    prims_sum_row_kernel<<<grid, block>>>(input, output, num_full_tiles, tail_elements, zero, tile_size);
    cudaDeviceSynchronize();
}