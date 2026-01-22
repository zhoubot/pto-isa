// PTO Program: rmsnorm_tile
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rmsnorm_tile
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 82,304 bytes (80.4 KB)
//   Total capacity (w/ reuse): 49,408 bytes (48.2 KB)
//   Reuse savings:            32,896 bytes (40.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   gamma                32x128     f32     16384   [  1,  10]           -
//   result               32x128     f32     16384   [ 10,  11]           <- x
//   row_mean             32x1       f32       128   [  5,   8]           -
//   row_rsqrt            32x1       f32       128   [  8,   9]           <- row_sum
//   row_sum              32x1       f32       128   [  3,   5]           -
//   x                    32x128     f32     16384   [  0,   9]           -
//   x_norm               32x128     f32     16384   [  9,  10]           <- x_sq
//   x_sq                 32x128     f32     16384   [  2,   3]           -
//
// BUFFER REUSE MAP:
//   row_rsqrt reuses buffer of row_sum
//   x_norm reuses buffer of x_sq
//   result reuses buffer of x
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void rmsnorm_tile(float* input, float* weights, float* output) {
    float x[32][128];
    float x_sq[32][128];
    float row_sum[32][1];
    float row_mean[32][1];
    float row_rsqrt[32][1];
    float x_norm[32][128];
    float gamma[32][128];
    float result[32][128];

    // Loop fusion: 5 loop overheads saved

    // FUSED LOOP (3 ops): x=TLOAD(input,0,0); gamma=TLOAD(weights,0,0); x_sq=TMUL(x,x)
    for (int _row = 0; _row < 32; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 128 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&weights[_row * 128 + _col]);
            vst1q_f32(&gamma[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr4 = vmulq_f32(_v2, _v3);
            vst1q_f32(&x_sq[_row][_col], _vr4);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            x[_row][_col] = input[_row * 128 + _col];
            gamma[_row][_col] = weights[_row * 128 + _col];
            x_sq[_row][_col] = x[_row][_col] * x[_row][_col];
        }
    }

    // TROWSUM: row_sum = rowsum(x_sq)
    for (int _row = 0; _row < 32; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 128; _col++) {
            _sum += x_sq[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    int inv_cols = 0.0078125;

    // FUSED LOOP (1 ops): row_mean=TMULS(row_sum,inv_colsf)
    float32x4_t _vs5 = vdupq_n_f32(inv_colsf);
    for (int _row = 0; _row < 32; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v6 = vld1q_f32(&row_sum[_row][_col]);
            float32x4_t _vr7 = vmulq_f32(_v6, _vs5);
            vst1q_f32(&row_mean[_row][_col], _vr7);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            row_mean[_row][_col] = row_sum[_row][_col] * inv_colsf;
        }
    }

    int eps = 1e-05;

    // FUSED LOOP (2 ops): row_mean=TADDS(row_mean,epsf); row_rsqrt=TRSQRT(row_mean)
    float32x4_t _vs8 = vdupq_n_f32(epsf);
    for (int _row = 0; _row < 32; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v9 = vld1q_f32(&row_mean[_row][_col]);
            float32x4_t _vr10 = vaddq_f32(_v9, _vs8);
            vst1q_f32(&row_mean[_row][_col], _vr10);
            float32x4_t _v11 = vld1q_f32(&row_mean[_row][_col]);
            float32x4_t _vr12 = vrsqrteq_f32(_v11);
            vst1q_f32(&row_rsqrt[_row][_col], _vr12);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            row_mean[_row][_col] = row_mean[_row][_col] + epsf;
            row_rsqrt[_row][_col] = 1.0f / sqrtf(row_mean[_row][_col]);
        }
    }

    // FUSED LOOP (3 ops): x_norm=TROWEXPANDMUL(x,row_rsqrt); result=TMUL(x_norm,gamma); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 32; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _v013 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vb15 = vdupq_n_f32(row_rsqrt[_row][0]);
            float32x4_t _vr14 = vmulq_f32(_v013, _vb15);
            vst1q_f32(&x_norm[_row][_col], _vr14);
            float32x4_t _v16 = vld1q_f32(&x_norm[_row][_col]);
            float32x4_t _v17 = vld1q_f32(&gamma[_row][_col]);
            float32x4_t _vr18 = vmulq_f32(_v16, _v17);
            vst1q_f32(&result[_row][_col], _vr18);
            float32x4_t _vs19 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs19);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            x_norm[_row][_col] = x[_row][_col] * row_rsqrt[_row][0];
            result[_row][_col] = x_norm[_row][_col] * gamma[_row][_col];
            output[_row * 128 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "rmsnorm_tile"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input",
    "weights",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(16384),
    (size_t)(16384),
    (size_t)(16384),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
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
    rmsnorm_tile((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER