// PTO Program: F_selu
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_selu
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     9
//   Total capacity (no reuse): 2,304 bytes (2.2 KB)
//   Total capacity (w/ reuse): 1,024 bytes (1.0 KB)
//   Reuse savings:            1,280 bytes (55.6%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   alpha_scaled         8x8        f32       256   [  4,   6]           <- exp_x
//   elu_result           8x8        f32       256   [  7,   8]           <- alpha_scaled
//   exp_minus_one        8x8        f32       256   [  3,   4]           <- x
//   exp_x                8x8        f32       256   [  2,   3]           -
//   neg_part             8x8        f32       256   [  6,   7]           -
//   pos_part             8x8        f32       256   [  1,   7]           -
//   result               8x8        f32       256   [  8,   9]           <- pos_part
//   x                    8x8        f32       256   [  0,   2]           -
//   zeros                8x8        f32       256   [  5,   6]           <- exp_minus_one
//
// BUFFER REUSE MAP:
//   exp_minus_one reuses buffer of x
//   alpha_scaled reuses buffer of exp_x
//   zeros reuses buffer of exp_minus_one
//   elu_result reuses buffer of alpha_scaled
//   result reuses buffer of pos_part
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

__device__ float x[8][8];
__device__ float pos_part[8][8];
__device__ float exp_x[8][8];
__device__ float exp_minus_one[8][8];
__device__ float alpha_scaled[8][8];
__device__ float zeros[8][8];
__device__ float neg_part[8][8];
__device__ float elu_result[8][8];
__device__ float result[8][8];

__global__ void F_selu_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 9 loop overheads saved

    // FUSED (10 ops): x=TLOAD(...); pos_part=TRELU(...); exp_x=TEXP(...); exp_minus_one=TADDS(...); alpha_scaled=TMULS(...); zeros=TEXPANDS(...); neg_part=TMIN(...); elu_result=TADD(...); result=TMULS(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        pos_part[_row][_col] = fmaxf(x[_row][_col], 0.0f);
        exp_x[_row][_col] = __expf(x[_row][_col]);
        exp_minus_one[_row][_col] = exp_x[_row][_col] + -1.0f;
        alpha_scaled[_row][_col] = exp_minus_one[_row][_col] * 1.6732632423543772f;
        zeros[_row][_col] = 0.0f;
        neg_part[_row][_col] = fminf(alpha_scaled[_row][_col], zeros[_row][_col]);
        elu_result[_row][_col] = pos_part[_row][_col] + neg_part[_row][_col];
        result[_row][_col] = elu_result[_row][_col] * 1.0507009873554805f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_selu(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_selu_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}