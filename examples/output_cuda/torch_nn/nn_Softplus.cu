// PTO Program: nn_Softplus
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_Softplus
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
//   exp_x                8x8        f32       256   [  2,   3]           <- x
//   log_out              8x8        f32       256   [  4,   5]           <- exp_x
//   one_plus             8x8        f32       256   [  3,   4]           <- scaled_x
//   result               8x8        f32       256   [  5,   6]           <- one_plus
//   scaled_x             8x8        f32       256   [  1,   2]           -
//   x                    8x8        f32       256   [  0,   1]           -
//
// BUFFER REUSE MAP:
//   exp_x reuses buffer of x
//   one_plus reuses buffer of scaled_x
//   log_out reuses buffer of exp_x
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
__device__ float scaled_x[8][8];
__device__ float exp_x[8][8];
__device__ float one_plus[8][8];
__device__ float log_out[8][8];
__device__ float result[8][8];

__global__ void nn_Softplus_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 6 loop overheads saved

    // FUSED (7 ops): x=TLOAD(...); scaled_x=TMULS(...); exp_x=TEXP(...); one_plus=TADDS(...); log_out=TLOG(...); result=TDIVS(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        scaled_x[_row][_col] = x[_row][_col] * 1.0f;
        exp_x[_row][_col] = __expf(scaled_x[_row][_col]);
        one_plus[_row][_col] = exp_x[_row][_col] + 1.0f;
        log_out[_row][_col] = __logf(one_plus[_row][_col]);
        result[_row][_col] = log_out[_row][_col] / 1.0f;
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_Softplus(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_Softplus_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}