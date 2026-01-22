// PTO Program: nn_Embedding
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_Embedding
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 4,352 bytes (4.2 KB)
//   Total capacity (w/ reuse): 4,352 bytes (4.2 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   indices_onehot       8x64       f32      2048   [  0,  -1]           -
//   result               8x8        f32       256   [  2,   3]           -
//   weight               64x8       f32      2048   [  1,  -1]           -
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

__device__ float indices_onehot[8][64];
__device__ float weight[64][8];
__device__ float result[8][8];

__global__ void nn_Embedding_kernel(float* indices_mem, float* weight_mem, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 0 loop overheads saved

    // FUSED (1 ops): indices_onehot=TLOAD(...)
    if (_row < 8 && _col < 64) {
        indices_onehot[_row][_col] = indices_mem[_row * 64 + _col];
    }

    // FUSED (1 ops): weight=TLOAD(...)
    if (_row < 64 && _col < 8) {
        weight[_row][_col] = weight_mem[_row * 8 + _col];
    }

    // TMATMUL: result = indices_onehot @ weight
    if (_row < 8 && _col < 8) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 64; _k++) _sum += indices_onehot[_row][_k] * weight[_k][_col];
        result[_row][_col] = _sum;}

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_Embedding(float* indices_mem, float* weight_mem, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_Embedding_kernel<<<grid, block>>>(indices_mem, weight_mem, output);
    cudaDeviceSynchronize();
}