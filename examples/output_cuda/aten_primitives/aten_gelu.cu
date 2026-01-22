// PTO Program: aten_gelu
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: aten_gelu
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 114,688 bytes (112.0 KB)
//   Total capacity (w/ reuse): 114,688 bytes (112.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg              1x4096     f32     16384   [  6,  18]           -
//   neg_scaled           1x4096     f32     16384   [  5,  17]           -
//   one_plus             1x4096     f32     16384   [  7,  19]           -
//   result               1x4096     f32     16384   [  9,  21]           -
//   scaled_x             1x4096     f32     16384   [  4,  16]           -
//   sigmoid_out          1x4096     f32     16384   [  8,  20]           -
//   x                    1x4096     f32     16384   [  3,  20]           -
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
__device__ float scaled_x[1][4096];
__device__ float neg_scaled[1][4096];
__device__ float exp_neg[1][4096];
__device__ float one_plus[1][4096];
__device__ float sigmoid_out[1][4096];
__device__ float result[1][4096];

__global__ void aten_gelu_kernel(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 14 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED (8 ops): x=TLOAD(...); scaled_x=TMULS(...); neg_scaled=TNEG(...); exp_neg=TEXP(...); one_plus=TADDS(...); sigmoid_out=TRECIP(...); result=TMUL(...); output=TSTORE(...)
        if (_row < 1 && _col < 4096) {
            x[_row][_col] = input[(tile_idx) * 4096 + _row * 4096 + _col];
            scaled_x[_row][_col] = x[_row][_col] * 1.702f;
            neg_scaled[_row][_col] = -scaled_x[_row][_col];
            exp_neg[_row][_col] = __expf(neg_scaled[_row][_col]);
            one_plus[_row][_col] = exp_neg[_row][_col] + 1.0f;
            sigmoid_out[_row][_col] = 1.0f / one_plus[_row][_col];
            result[_row][_col] = x[_row][_col] * sigmoid_out[_row][_col];
            output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED (8 ops): x=TLOAD(...); scaled_x=TMULS(...); neg_scaled=TNEG(...); exp_neg=TEXP(...); one_plus=TADDS(...); sigmoid_out=TRECIP(...); result=TMUL(...); output=TSTORE(...)
        if (_row < 1 && _col < 4096) {
            x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 4096 + _col];
            scaled_x[_row][_col] = x[_row][_col] * 1.702f;
            neg_scaled[_row][_col] = -scaled_x[_row][_col];
            exp_neg[_row][_col] = __expf(neg_scaled[_row][_col]);
            one_plus[_row][_col] = exp_neg[_row][_col] + 1.0f;
            sigmoid_out[_row][_col] = 1.0f / one_plus[_row][_col];
            result[_row][_col] = x[_row][_col] * sigmoid_out[_row][_col];
            output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
        }

    }

}

void aten_gelu(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    aten_gelu_kernel<<<grid, block>>>(input, output, num_full_tiles, tail_elements, zero, tile_size);
    cudaDeviceSynchronize();
}