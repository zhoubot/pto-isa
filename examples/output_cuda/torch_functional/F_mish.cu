// PTO Program: F_mish
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_mish
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
//   exp_2sp              8x8        f32       256   [  5,   7]           <- softplus
//   exp_x                8x8        f32       256   [  1,   2]           -
//   one_plus_exp         8x8        f32       256   [  2,   3]           -
//   result               8x8        f32       256   [  9,  10]           <- tanh_num
//   softplus             8x8        f32       256   [  3,   4]           <- exp_x
//   sp_2                 8x8        f32       256   [  4,   5]           <- one_plus_exp
//   tanh_den             8x8        f32       256   [  7,   8]           -
//   tanh_num             8x8        f32       256   [  6,   8]           <- sp_2
//   tanh_out             8x8        f32       256   [  8,   9]           <- exp_2sp
//   x                    8x8        f32       256   [  0,   9]           -
//
// BUFFER REUSE MAP:
//   softplus reuses buffer of exp_x
//   sp_2 reuses buffer of one_plus_exp
//   exp_2sp reuses buffer of softplus
//   tanh_num reuses buffer of sp_2
//   tanh_out reuses buffer of exp_2sp
//   result reuses buffer of tanh_num
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
__device__ float sp_2[8][8];
__device__ float exp_2sp[8][8];
__device__ float tanh_num[8][8];
__device__ float tanh_den[8][8];
__device__ float tanh_out[8][8];
__device__ float result[8][8];

__global__ void F_mish_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 10 loop overheads saved

    // FUSED (11 ops): x=TLOAD(...); exp_x=TEXP(...); one_plus_exp=TADDS(...); softplus=TLOG(...); sp_2=TMULS(...); exp_2sp=TEXP(...); tanh_num=TADDS(...); tanh_den=TADDS(...); tanh_out=TDIV(...); result=TMUL(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        exp_x[_row][_col] = __expf(x[_row][_col]);
        one_plus_exp[_row][_col] = exp_x[_row][_col] + 1.0f;
        softplus[_row][_col] = __logf(one_plus_exp[_row][_col]);
        sp_2[_row][_col] = softplus[_row][_col] * 2.0f;
        exp_2sp[_row][_col] = __expf(sp_2[_row][_col]);
        tanh_num[_row][_col] = exp_2sp[_row][_col] + -1.0f;
        tanh_den[_row][_col] = exp_2sp[_row][_col] + 1.0f;
        tanh_out[_row][_col] = tanh_num[_row][_col] / tanh_den[_row][_col];
        result[_row][_col] = x[_row][_col] * tanh_out[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_mish(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_mish_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}