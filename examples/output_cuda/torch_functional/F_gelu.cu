// PTO Program: F_gelu
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_gelu
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     14
//   Total capacity (no reuse): 3,584 bytes (3.5 KB)
//   Total capacity (w/ reuse): 1,280 bytes (1.2 KB)
//   Reuse savings:            2,304 bytes (64.3%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   coeff_x3             8x8        f32       256   [  3,   4]           <- x_sq
//   cosh_approx          8x8        f32       256   [  9,  10]           -
//   exp_neg              8x8        f32       256   [-, -]               -
//   exp_pos              8x8        f32       256   [  7,   9]           <- inner
//   half_x               8x8        f32       256   [ 12,  13]           <- cosh_approx
//   inner                8x8        f32       256   [  4,   5]           <- x_cubed
//   one_plus             8x8        f32       256   [ 11,  13]           <- sinh_approx
//   result               8x8        f32       256   [ 13,  14]           <- x
//   scaled               8x8        f32       256   [  5,   7]           <- coeff_x3
//   sinh_approx          8x8        f32       256   [  8,  10]           <- scaled
//   tanh_out             8x8        f32       256   [ 10,  11]           <- exp_pos
//   x                    8x8        f32       256   [  0,  12]           -
//   x_cubed              8x8        f32       256   [  2,   3]           -
//   x_sq                 8x8        f32       256   [  1,   2]           -
//
// BUFFER REUSE MAP:
//   coeff_x3 reuses buffer of x_sq
//   inner reuses buffer of x_cubed
//   scaled reuses buffer of coeff_x3
//   tanh_out reuses buffer of exp_pos
//   exp_pos reuses buffer of inner
//   sinh_approx reuses buffer of scaled
//   one_plus reuses buffer of sinh_approx
//   half_x reuses buffer of cosh_approx
//   result reuses buffer of x
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
__device__ float x_cubed[8][8];
__device__ float x_sq[8][8];
__device__ float coeff_x3[8][8];
__device__ float inner[8][8];
__device__ float scaled[8][8];
__device__ float tanh_out[8][8];
__device__ float exp_pos[8][8];
__device__ float exp_neg[8][8];
__device__ float sinh_approx[8][8];
__device__ float cosh_approx[8][8];
__device__ float one_plus[8][8];
__device__ float half_x[8][8];
__device__ float result[8][8];

__global__ void F_gelu_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 14 loop overheads saved

    // FUSED (15 ops): x=TLOAD(...); x_sq=TMUL(...); x_cubed=TMUL(...); coeff_x3=TMULS(...); inner=TADD(...); scaled=TMULS(...); scaled=TMULS(...); exp_pos=TEXP(...); sinh_approx=TADDS(...); cosh_approx=TADDS(...); tanh_out=TDIV(...); one_plus=TADDS(...); half_x=TMULS(...); result=TMUL(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        x_sq[_row][_col] = x[_row][_col] * x[_row][_col];
        x_cubed[_row][_col] = x_sq[_row][_col] * x[_row][_col];
        coeff_x3[_row][_col] = x_cubed[_row][_col] * 0.044715f;
        inner[_row][_col] = x[_row][_col] + coeff_x3[_row][_col];
        scaled[_row][_col] = inner[_row][_col] * 0.7978845608028654f;
        scaled[_row][_col] = scaled[_row][_col] * 2.0f;
        exp_pos[_row][_col] = __expf(scaled[_row][_col]);
        sinh_approx[_row][_col] = exp_pos[_row][_col] + -1.0f;
        cosh_approx[_row][_col] = exp_pos[_row][_col] + 1.0f;
        tanh_out[_row][_col] = sinh_approx[_row][_col] / cosh_approx[_row][_col];
        one_plus[_row][_col] = tanh_out[_row][_col] + 1.0f;
        half_x[_row][_col] = x[_row][_col] * 0.5f;
        result[_row][_col] = half_x[_row][_col] * one_plus[_row][_col];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_gelu(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_gelu_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}