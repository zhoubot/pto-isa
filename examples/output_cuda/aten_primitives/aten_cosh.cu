// PTO Program: aten_cosh
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: aten_cosh
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 98,304 bytes (96.0 KB)
//   Total capacity (w/ reuse): 98,304 bytes (96.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg_x            1x4096     f32     16384   [  6,  17]           -
//   exp_x                1x4096     f32     16384   [  4,  17]           -
//   neg_x                1x4096     f32     16384   [  5,  16]           -
//   result               1x4096     f32     16384   [  8,  19]           -
//   sum                  1x4096     f32     16384   [  7,  18]           -
//   x                    1x4096     f32     16384   [  3,  15]           -
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
__device__ float neg_x[1][4096];
__device__ float exp_x[1][4096];
__device__ float exp_neg_x[1][4096];
__device__ float sum[1][4096];
__device__ float result[1][4096];

__global__ void aten_cosh_kernel(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 12 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED (7 ops): x=TLOAD(...); exp_x=TEXP(...); neg_x=TNEG(...); exp_neg_x=TEXP(...); sum=TADD(...); result=TDIVS(...); output=TSTORE(...)
        if (_row < 1 && _col < 4096) {
            x[_row][_col] = input[(tile_idx) * 4096 + _row * 4096 + _col];
            exp_x[_row][_col] = __expf(x[_row][_col]);
            neg_x[_row][_col] = -x[_row][_col];
            exp_neg_x[_row][_col] = __expf(neg_x[_row][_col]);
            sum[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
            result[_row][_col] = sum[_row][_col] / 2.0f;
            output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED (7 ops): x=TLOAD(...); exp_x=TEXP(...); neg_x=TNEG(...); exp_neg_x=TEXP(...); sum=TADD(...); result=TDIVS(...); output=TSTORE(...)
        if (_row < 1 && _col < 4096) {
            x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 4096 + _col];
            exp_x[_row][_col] = __expf(x[_row][_col]);
            neg_x[_row][_col] = -x[_row][_col];
            exp_neg_x[_row][_col] = __expf(neg_x[_row][_col]);
            sum[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
            result[_row][_col] = sum[_row][_col] / 2.0f;
            output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
        }

    }

}

void aten_cosh(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    aten_cosh_kernel<<<grid, block>>>(input, output, num_full_tiles, tail_elements, zero, tile_size);
    cudaDeviceSynchronize();
}