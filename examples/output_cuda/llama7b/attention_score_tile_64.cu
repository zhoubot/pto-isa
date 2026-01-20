// PTO Program: attention_score_tile_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: attention_score_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 163,840 bytes (160.0 KB)
//   Total capacity (w/ reuse): 163,840 bytes (160.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   k_t                  128x128    f32     65536   [  1,  -1]           -
//   q                    64x128     f32     32768   [  0,  -1]           -
//   scaled_scores        64x128     f32     32768   [  4,   5]           -
//   scores               64x128     f32     32768   [  2,   4]           -
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

__device__ float q[64][128];
__device__ float k_t[128][128];
__device__ float scores[64][128];
__device__ float scaled_scores[64][128];

__global__ void attention_score_tile_64_kernel(float* input_q, float* input_kt, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 1 loop overheads saved

    // FUSED (1 ops): q=TLOAD(...)
    if (_row < 64 && _col < 128) {
        q[_row][_col] = input_q[_row * 128 + _col];
    }

    // FUSED (1 ops): k_t=TLOAD(...)
    if (_row < 128 && _col < 128) {
        k_t[_row][_col] = input_kt[_row * 128 + _col];
    }

    // TMATMUL: scores = q @ k_t
    if (_row < 64 && _col < 128) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 128; _k++) _sum += q[_row][_k] * k_t[_k][_col];
        scores[_row][_col] = _sum;}

    int scale = 0.08838834764831843;

    // FUSED (2 ops): scaled_scores=TMULS(...); output=TSTORE(...)
    if (_row < 64 && _col < 128) {
        scaled_scores[_row][_col] = scores[_row][_col] * scalef;
        output[_row * 128 + _col] = scaled_scores[_row][_col];
    }

}

void attention_score_tile_64(float* input_q, float* input_kt, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    attention_score_tile_64_kernel<<<grid, block>>>(input_q, input_kt, output, scale);
    cudaDeviceSynchronize();
}