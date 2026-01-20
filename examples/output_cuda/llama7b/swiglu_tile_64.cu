// PTO Program: swiglu_tile_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: swiglu_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 262,144 bytes (256.0 KB)
//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)
//   Reuse savings:            131,072 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg_gate         64x128     f32     32768   [  3,   4]           -
//   gate                 64x128     f32     32768   [  0,   6]           -
//   gate_silu            64x128     f32     32768   [  6,   7]           <- one_plus_exp
//   neg_gate             64x128     f32     32768   [  2,   3]           -
//   one_plus_exp         64x128     f32     32768   [  4,   5]           <- neg_gate
//   result               64x128     f32     32768   [  7,   8]           <- gate
//   sigmoid_gate         64x128     f32     32768   [  5,   6]           <- exp_neg_gate
//   up                   64x128     f32     32768   [  1,   7]           -
//
// BUFFER REUSE MAP:
//   one_plus_exp reuses buffer of neg_gate
//   sigmoid_gate reuses buffer of exp_neg_gate
//   gate_silu reuses buffer of one_plus_exp
//   result reuses buffer of gate
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

__device__ float gate[64][128];
__device__ float up[64][128];
__device__ float neg_gate[64][128];
__device__ float exp_neg_gate[64][128];
__device__ float one_plus_exp[64][128];
__device__ float sigmoid_gate[64][128];
__device__ float gate_silu[64][128];
__device__ float result[64][128];

__global__ void swiglu_tile_64_kernel(float* input_gate, float* input_up, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 8 loop overheads saved

    // FUSED (9 ops): gate=TLOAD(...); up=TLOAD(...); neg_gate=TNEG(...); exp_neg_gate=TEXP(...); one_plus_exp=TADDS(...); sigmoid_gate=TRECIP(...); gate_silu=TMUL(...); result=TMUL(...); output=TSTORE(...)
    if (_row < 64 && _col < 128) {
        gate[_row][_col] = input_gate[_row * 128 + _col];
        up[_row][_col] = input_up[_row * 128 + _col];
        neg_gate[_row][_col] = -gate[_row][_col];
        exp_neg_gate[_row][_col] = __expf(neg_gate[_row][_col]);
        one_plus_exp[_row][_col] = exp_neg_gate[_row][_col] + 1.0f;
        sigmoid_gate[_row][_col] = 1.0f / one_plus_exp[_row][_col];
        gate_silu[_row][_col] = gate[_row][_col] * sigmoid_gate[_row][_col];
        result[_row][_col] = gate_silu[_row][_col] * up[_row][_col];
        output[_row * 128 + _col] = result[_row][_col];
    }

}

void swiglu_tile_64(float* input_gate, float* input_up, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    swiglu_tile_64_kernel<<<grid, block>>>(input_gate, input_up, output);
    cudaDeviceSynchronize();
}