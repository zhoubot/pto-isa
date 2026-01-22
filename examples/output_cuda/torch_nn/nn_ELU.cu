// PTO Program: nn_ELU
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_ELU
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     10
//   Total capacity (no reuse): 2,560 bytes (2.5 KB)
//   Total capacity (w/ reuse): 1,024 bytes (1.0 KB)
//   Reuse savings:            1,536 bytes (60.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_minus_1          8x8        f32       256   [  3,   4]           <- x
//   exp_x                8x8        f32       256   [  2,   3]           -
//   neg_contrib          8x8        f32       256   [  7,   8]           <- neg_x
//   neg_part             8x8        f32       256   [-, -]               -
//   neg_relu             8x8        f32       256   [  6,   7]           <- scaled
//   neg_x                8x8        f32       256   [  5,   6]           <- exp_minus_1
//   pos_part             8x8        f32       256   [  1,   8]           -
//   result               8x8        f32       256   [  8,   9]           <- neg_relu
//   scaled               8x8        f32       256   [  4,   5]           <- exp_x
//   x                    8x8        f32       256   [  0,   2]           -
//
// BUFFER REUSE MAP:
//   exp_minus_1 reuses buffer of x
//   scaled reuses buffer of exp_x
//   neg_x reuses buffer of exp_minus_1
//   neg_relu reuses buffer of scaled
//   neg_contrib reuses buffer of neg_x
//   result reuses buffer of neg_relu
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
__device__ float exp_minus_1[8][8];
__device__ float scaled[8][8];
__device__ float neg_x[8][8];
__device__ float neg_relu[8][8];
__device__ float neg_part[8][8];
__device__ float neg_contrib[8][8];
__device__ float result[8][8];

__global__ void nn_ELU_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 9 loop overheads saved

    // FUSED (10 ops): x=TLOAD(...); pos_part=TRELU(...); exp_x=TEXP(...); exp_minus_1=TADDS(...); scaled=TMULS(...); neg_x=TNEG(...); neg_relu=TRELU(...); neg_contrib=TNEG(...); result=TADD(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        pos_part[_row][_col] = fmaxf(x[_row][_col], 0.0f);
        exp_x[_row][_col] = __expf(x[_row][_col]);
        exp_minus_1[_row][_col] = exp_x[_row][_col] + -1.0f;
        scaled[_row][_col] = exp_minus_1[_row][_col] * 1.0f;
        neg_x[_row][_col] = -scaled[_row][_col];
        neg_relu[_row][_col] = fmaxf(neg_x[_row][_col], 0.0f);
        neg_contrib[_row][_col] = -neg_relu[_row][_col];
        result[_row][_col] = pos_part[_row][_col] + neg_contrib[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_ELU(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_ELU_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}