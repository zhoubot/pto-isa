// PTO Program: nn_Mish
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_Mish
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     11
//   Total capacity (no reuse): 2,816 bytes (2.8 KB)
//   Total capacity (w/ reuse): 1,280 bytes (1.2 KB)
//   Reuse savings:            1,536 bytes (54.5%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg_sp           8x8        f32       256   [  6,   8]           <- softplus
//   exp_sp               8x8        f32       256   [  4,   8]           <- one_plus_exp
//   exp_x                8x8        f32       256   [  1,   2]           -
//   neg_sp               8x8        f32       256   [  5,   6]           -
//   one_plus_exp         8x8        f32       256   [  2,   3]           -
//   result               8x8        f32       256   [ 10,  11]           <- exp_neg_sp
//   softplus             8x8        f32       256   [  3,   5]           <- exp_x
//   tanh_den             8x8        f32       256   [  8,   9]           -
//   tanh_num             8x8        f32       256   [  7,   9]           <- neg_sp
//   tanh_out             8x8        f32       256   [  9,  10]           <- exp_sp
//   x                    8x8        f32       256   [  0,  10]           -
//
// BUFFER REUSE MAP:
//   softplus reuses buffer of exp_x
//   exp_sp reuses buffer of one_plus_exp
//   exp_neg_sp reuses buffer of softplus
//   tanh_num reuses buffer of neg_sp
//   tanh_out reuses buffer of exp_sp
//   result reuses buffer of exp_neg_sp
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
__device__ float exp_x[8][8];
__device__ float one_plus_exp[8][8];
__device__ float softplus[8][8];
__device__ float exp_sp[8][8];
__device__ float neg_sp[8][8];
__device__ float exp_neg_sp[8][8];
__device__ float tanh_num[8][8];
__device__ float tanh_den[8][8];
__device__ float tanh_out[8][8];
__device__ float result[8][8];

__global__ void nn_Mish_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 11 loop overheads saved

    // FUSED (12 ops): x=TLOAD(...); exp_x=TEXP(...); one_plus_exp=TADDS(...); softplus=TLOG(...); exp_sp=TEXP(...); neg_sp=TNEG(...); exp_neg_sp=TEXP(...); tanh_num=TSUB(...); tanh_den=TADD(...); tanh_out=TDIV(...); result=TMUL(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        exp_x[_row][_col] = __expf(x[_row][_col]);
        one_plus_exp[_row][_col] = exp_x[_row][_col] + 1.0f;
        softplus[_row][_col] = __logf(one_plus_exp[_row][_col]);
        exp_sp[_row][_col] = __expf(softplus[_row][_col]);
        neg_sp[_row][_col] = -softplus[_row][_col];
        exp_neg_sp[_row][_col] = __expf(neg_sp[_row][_col]);
        tanh_num[_row][_col] = exp_sp[_row][_col] - exp_neg_sp[_row][_col];
        tanh_den[_row][_col] = exp_sp[_row][_col] + exp_neg_sp[_row][_col];
        tanh_out[_row][_col] = tanh_num[_row][_col] / tanh_den[_row][_col];
        result[_row][_col] = x[_row][_col] * tanh_out[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_Mish(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_Mish_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}