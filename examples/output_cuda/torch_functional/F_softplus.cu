// PTO Program: F_softplus
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_softplus
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 1,536 bytes (1.5 KB)
//   Total capacity (w/ reuse): 512 bytes (0.5 KB)
//   Reuse savings:            1,024 bytes (66.7%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   beta_x               8x8        f32       256   [  1,   2]           -
//   exp_bx               8x8        f32       256   [  2,   3]           <- x
//   log_val              8x8        f32       256   [  4,   5]           <- exp_bx
//   one_plus             8x8        f32       256   [  3,   4]           <- beta_x
//   result               8x8        f32       256   [  5,   6]           <- one_plus
//   x                    8x8        f32       256   [  0,   1]           -
//
// BUFFER REUSE MAP:
//   exp_bx reuses buffer of x
//   one_plus reuses buffer of beta_x
//   log_val reuses buffer of exp_bx
//   result reuses buffer of one_plus
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
__device__ float beta_x[8][8];
__device__ float exp_bx[8][8];
__device__ float one_plus[8][8];
__device__ float log_val[8][8];
__device__ float result[8][8];

__global__ void F_softplus_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 6 loop overheads saved

    // FUSED (7 ops): x=TLOAD(...); beta_x=TMULS(...); exp_bx=TEXP(...); one_plus=TADDS(...); log_val=TLOG(...); result=TDIVS(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        beta_x[_row][_col] = x[_row][_col] * 1.0f;
        exp_bx[_row][_col] = __expf(beta_x[_row][_col]);
        one_plus[_row][_col] = exp_bx[_row][_col] + 1.0f;
        log_val[_row][_col] = __logf(one_plus[_row][_col]);
        result[_row][_col] = log_val[_row][_col] / 1.0f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void F_softplus(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_softplus_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}