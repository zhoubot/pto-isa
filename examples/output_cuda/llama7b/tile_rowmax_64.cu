// PTO Program: tile_rowmax_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_rowmax_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 33,024 bytes (32.2 KB)
//   Total capacity (w/ reuse): 33,024 bytes (32.2 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               64x1       f32       256   [  1,   2]           -
//   x                    64x128     f32     32768   [  0,   1]           -
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
__device__ float result[64][1];

__global__ void tile_rowmax_64_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 0 loop overheads saved

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 64 && _col < 128) {
        x[_row][_col] = input[_row * 128 + _col];
    }

    // TROWMAX: result = rowmax(x)
    if (_col == 0 && _row < 64) {
        float _max = x[_row][0];
        for (int _c = 1; _c < 128; _c++) if (x[_row][_c] > _max) _max = x[_row][_c];
        result[_row][0] = _max;}

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 64 && _col < 1) {
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void tile_rowmax_64(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tile_rowmax_64_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}