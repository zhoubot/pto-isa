// PTO Program: tile_silu_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_silu_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 196,608 bytes (192.0 KB)
//   Total capacity (w/ reuse): 98,304 bytes (96.0 KB)
//   Reuse savings:            98,304 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg_x            64x128     f32     32768   [  2,   3]           -
//   neg_x                64x128     f32     32768   [  1,   2]           -
//   one_plus_exp         64x128     f32     32768   [  3,   4]           <- neg_x
//   result               64x128     f32     32768   [  5,   6]           <- one_plus_exp
//   sigmoid              64x128     f32     32768   [  4,   5]           <- exp_neg_x
//   x                    64x128     f32     32768   [  0,   5]           -
//
// BUFFER REUSE MAP:
//   one_plus_exp reuses buffer of neg_x
//   sigmoid reuses buffer of exp_neg_x
//   result reuses buffer of one_plus_exp
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
__device__ float neg_x[64][128];
__device__ float exp_neg_x[64][128];
__device__ float one_plus_exp[64][128];
__device__ float sigmoid[64][128];
__device__ float result[64][128];

__global__ void tile_silu_64_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 6 loop overheads saved

    // FUSED (7 ops): x=TLOAD(...); neg_x=TNEG(...); exp_neg_x=TEXP(...); one_plus_exp=TADDS(...); sigmoid=TRECIP(...); result=TMUL(...); output=TSTORE(...)
    if (_row < 64 && _col < 128) {
        x[_row][_col] = input[_row * 128 + _col];
        neg_x[_row][_col] = -x[_row][_col];
        exp_neg_x[_row][_col] = __expf(neg_x[_row][_col]);
        one_plus_exp[_row][_col] = exp_neg_x[_row][_col] + 1.0f;
        sigmoid[_row][_col] = 1.0f / one_plus_exp[_row][_col];
        result[_row][_col] = x[_row][_col] * sigmoid[_row][_col];
        output[_row * 128 + _col] = result[_row][_col];
    }

}

void tile_silu_64(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tile_silu_64_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}