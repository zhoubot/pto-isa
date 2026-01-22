// PTO Program: linear_projection_qkv
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: linear_projection_qkv
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 8,960 bytes (8.8 KB)
//   Total capacity (w/ reuse): 8,960 bytes (8.8 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   K                    8x8        f32       256   [  5,   8]           -
//   Q                    8x8        f32       256   [  4,   7]           -
//   V                    8x8        f32       256   [  6,   9]           -
//   W_K                  64x8       f32      2048   [  2,  -1]           -
//   W_Q                  64x8       f32      2048   [  1,  -1]           -
//   W_V                  64x8       f32      2048   [  3,  -1]           -
//   X                    8x64       f32      2048   [  0,  -1]           -
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

__device__ float X[8][64];
__device__ float W_Q[64][8];
__device__ float W_K[64][8];
__device__ float W_V[64][8];
__device__ float Q[8][8];
__device__ float K[8][8];
__device__ float V[8][8];

__global__ void linear_projection_qkv_kernel(float* X_mem, float* WQ_mem, float* WK_mem, float* WV_mem, float* Q_mem, float* K_mem, float* V_mem) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (1 ops): X=TLOAD(...)
    if (_row < 8 && _col < 64) {
        X[_row][_col] = X_mem[_row * 64 + _col];
    }

    // FUSED (3 ops): W_Q=TLOAD(...); W_K=TLOAD(...); W_V=TLOAD(...)
    if (_row < 64 && _col < 8) {
        W_Q[_row][_col] = WQ_mem[_row * 8 + _col];
        W_K[_row][_col] = WK_mem[_row * 8 + _col];
        W_V[_row][_col] = WV_mem[_row * 8 + _col];
    }

    // TMATMUL: Q = X @ W_Q
    if (_row < 8 && _col < 8) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 64; _k++) _sum += X[_row][_k] * W_Q[_k][_col];
        Q[_row][_col] = _sum;}

    // TMATMUL: K = X @ W_K
    if (_row < 8 && _col < 8) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 64; _k++) _sum += X[_row][_k] * W_K[_k][_col];
        K[_row][_col] = _sum;}

    // TMATMUL: V = X @ W_V
    if (_row < 8 && _col < 8) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 64; _k++) _sum += X[_row][_k] * W_V[_k][_col];
        V[_row][_col] = _sum;}

    // FUSED (3 ops): Q_mem=TSTORE(...); K_mem=TSTORE(...); V_mem=TSTORE(...)
    if (_row < 8 && _col < 8) {
        Q_mem[_row * 8 + _col] = Q[_row][_col];
        K_mem[_row * 8 + _col] = K[_row][_col];
        V_mem[_row * 8 + _col] = V[_row][_col];
    }

}

void linear_projection_qkv(float* X_mem, float* WQ_mem, float* WK_mem, float* WV_mem, float* Q_mem, float* K_mem, float* V_mem) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    linear_projection_qkv_kernel<<<grid, block>>>(X_mem, WQ_mem, WK_mem, WV_mem, Q_mem, K_mem, V_mem);
    cudaDeviceSynchronize();
}