// PTO Program: aten_sigmoid
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: aten_sigmoid
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 81,920 bytes (80.0 KB)
//   Total capacity (w/ reuse): 81,920 bytes (80.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               1x4096     f32     16384   [  7,  17]           -
//   t1                   1x4096     f32     16384   [  4,  14]           -
//   t2                   1x4096     f32     16384   [  5,  15]           -
//   t3                   1x4096     f32     16384   [  6,  16]           -
//   x                    1x4096     f32     16384   [  3,  13]           -
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
__device__ float t1[1][4096];
__device__ float t2[1][4096];
__device__ float t3[1][4096];
__device__ float result[1][4096];

__global__ void aten_sigmoid_kernel(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 10 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED (6 ops): x=TLOAD(...); t1=TNEG(...); t2=TEXP(...); t3=TADDS(...); result=TRECIP(...); output=TSTORE(...)
        if (_row < 1 && _col < 4096) {
            x[_row][_col] = input[(tile_idx) * 4096 + _row * 4096 + _col];
            t1[_row][_col] = -x[_row][_col];
            t2[_row][_col] = __expf(t1[_row][_col]);
            t3[_row][_col] = t2[_row][_col] + 1.0f;
            result[_row][_col] = 1.0f / t3[_row][_col];
            output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED (6 ops): x=TLOAD(...); t1=TNEG(...); t2=TEXP(...); t3=TADDS(...); result=TRECIP(...); output=TSTORE(...)
        if (_row < 1 && _col < 4096) {
            x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 4096 + _col];
            t1[_row][_col] = -x[_row][_col];
            t2[_row][_col] = __expf(t1[_row][_col]);
            t3[_row][_col] = t2[_row][_col] + 1.0f;
            result[_row][_col] = 1.0f / t3[_row][_col];
            output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
        }

    }

}

void aten_sigmoid(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    aten_sigmoid_kernel<<<grid, block>>>(input, output, num_full_tiles, tail_elements, zero, tile_size);
    cudaDeviceSynchronize();
}