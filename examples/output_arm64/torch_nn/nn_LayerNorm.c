// PTO Program: nn_LayerNorm
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_LayerNorm
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     10
//   Total capacity (no reuse): 1,216 bytes (1.2 KB)
//   Total capacity (w/ reuse): 576 bytes (0.6 KB)
//   Reuse savings:            640 bytes (52.6%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   mean                 8x1        f32        32   [  2,   3]           -
//   result               8x8        f32       256   [  9,  10]           <- squared
//   row_sum              8x1        f32        32   [  1,   2]           -
//   squared              8x8        f32       256   [  4,   5]           <- x
//   std                  8x1        f32        32   [  8,   9]           <- variance
//   var_eps              8x1        f32        32   [  7,   8]           <- var_sum
//   var_sum              8x1        f32        32   [  5,   6]           <- row_sum
//   variance             8x1        f32        32   [  6,   7]           <- mean
//   x                    8x8        f32       256   [  0,   3]           -
//   x_minus_mean         8x8        f32       256   [  3,   9]           -
//
// BUFFER REUSE MAP:
//   squared reuses buffer of x
//   var_sum reuses buffer of row_sum
//   variance reuses buffer of mean
//   var_eps reuses buffer of var_sum
//   std reuses buffer of variance
//   result reuses buffer of squared
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void nn_LayerNorm(float* input, float* output) {
    float x[8][8];
    float row_sum[8][1];
    float mean[8][1];
    float x_minus_mean[8][8];
    float squared[8][8];
    float var_sum[8][1];
    float variance[8][1];
    float var_eps[8][1];
    float std[8][1];
    float result[8][8];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (1 ops): x=TLOAD(input,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
        }
    }

    // TROWSUM: row_sum = rowsum(x)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += x[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (1 ops): mean=TDIVS(row_sum,8.0f)
    float32x4_t _vs1 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v2 = vld1q_f32(&row_sum[_row][_col]);
            float32x4_t _vr3 = vdivq_f32(_v2, _vs1);
            vst1q_f32(&mean[_row][_col], _vr3);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            mean[_row][_col] = row_sum[_row][_col] / 8.0f;
        }
    }

    // FUSED LOOP (2 ops): x_minus_mean=TROWEXPANDSUB(x,mean); squared=TMUL(x_minus_mean,x_minus_mean)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v04 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vb6 = vdupq_n_f32(mean[_row][0]);
            float32x4_t _vr5 = vsubq_f32(_v04, _vb6);
            vst1q_f32(&x_minus_mean[_row][_col], _vr5);
            float32x4_t _v7 = vld1q_f32(&x_minus_mean[_row][_col]);
            float32x4_t _v8 = vld1q_f32(&x_minus_mean[_row][_col]);
            float32x4_t _vr9 = vmulq_f32(_v7, _v8);
            vst1q_f32(&squared[_row][_col], _vr9);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x_minus_mean[_row][_col] = x[_row][_col] - mean[_row][0];
            squared[_row][_col] = x_minus_mean[_row][_col] * x_minus_mean[_row][_col];
        }
    }

    // TROWSUM: var_sum = rowsum(squared)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += squared[_row][_col];
        }
        var_sum[_row][0] = _sum;}

    // FUSED LOOP (3 ops): variance=TDIVS(var_sum,8.0f); var_eps=TADDS(variance,1e-05f); std=TSQRT(var_eps)
    float32x4_t _vs10 = vdupq_n_f32(8.0f);
    float32x4_t _vs11 = vdupq_n_f32(1e-05f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v12 = vld1q_f32(&var_sum[_row][_col]);
            float32x4_t _vr13 = vdivq_f32(_v12, _vs10);
            vst1q_f32(&variance[_row][_col], _vr13);
            float32x4_t _v14 = vld1q_f32(&variance[_row][_col]);
            float32x4_t _vr15 = vaddq_f32(_v14, _vs11);
            vst1q_f32(&var_eps[_row][_col], _vr15);
            float32x4_t _v16 = vld1q_f32(&var_eps[_row][_col]);
            float32x4_t _vr17 = vsqrtq_f32(_v16);
            vst1q_f32(&std[_row][_col], _vr17);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            variance[_row][_col] = var_sum[_row][_col] / 8.0f;
            var_eps[_row][_col] = variance[_row][_col] + 1e-05f;
            std[_row][_col] = sqrtf(var_eps[_row][_col]);
        }
    }

    // FUSED LOOP (2 ops): result=TROWEXPANDDIV(x_minus_mean,std); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v018 = vld1q_f32(&x_minus_mean[_row][_col]);
            float32x4_t _vb20 = vdupq_n_f32(std[_row][0]);
            float32x4_t _vr19 = vdivq_f32(_v018, _vb20);
            vst1q_f32(&result[_row][_col], _vr19);
            float32x4_t _vs21 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs21);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            result[_row][_col] = x_minus_mean[_row][_col] / std[_row][0];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "nn_LayerNorm"; }
enum { kPtoNumMemrefs = 2 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
    (size_t)(256),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    1,
};
int pto_num_memrefs() { return kPtoNumMemrefs; }
const char* pto_memref_name(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return "";
    return kPtoMemrefNames[idx];
}
size_t pto_memref_bytes(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefBytes[idx];
}
const char* pto_memref_dtype(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return "";
    return kPtoMemrefDtypes[idx];
}
size_t pto_memref_elem_bytes(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefElemBytes[idx];
}
int pto_memref_is_output(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefIsOutput[idx];
}
void pto_launch(void **args, void *stream) {
    (void)stream;
    nn_LayerNorm((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER