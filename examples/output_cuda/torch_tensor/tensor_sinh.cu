// PTO Program: tensor_sinh
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_sinh
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 1,536 bytes (1.5 KB)
//   Total capacity (w/ reuse): 768 bytes (0.8 KB)
//   Reuse savings:            768 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   diff                 8x8        f32       256   [  4,   5]           <- neg_x
//   exp_neg_x            8x8        f32       256   [  3,   4]           <- x
//   exp_x                8x8        f32       256   [  2,   4]           -
//   neg_x                8x8        f32       256   [  1,   3]           -
//   result               8x8        f32       256   [  5,   6]           <- exp_x
//   x                    8x8        f32       256   [  0,   2]           -
//
// BUFFER REUSE MAP:
//   exp_neg_x reuses buffer of x
//   diff reuses buffer of neg_x
//   result reuses buffer of exp_x
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
__device__ float neg_x[8][8];
__device__ float exp_x[8][8];
__device__ float exp_neg_x[8][8];
__device__ float diff[8][8];
__device__ float result[8][8];

__global__ void tensor_sinh_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 6 loop overheads saved

    // FUSED (7 ops): x=TLOAD(...); neg_x=TNEG(...); exp_x=TEXP(...); exp_neg_x=TEXP(...); diff=TSUB(...); result=TDIVS(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        neg_x[_row][_col] = -x[_row][_col];
        exp_x[_row][_col] = __expf(x[_row][_col]);
        exp_neg_x[_row][_col] = __expf(neg_x[_row][_col]);
        diff[_row][_col] = exp_x[_row][_col] - exp_neg_x[_row][_col];
        result[_row][_col] = diff[_row][_col] / 2.0f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void tensor_sinh(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tensor_sinh_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}