// PTO Program: flash_attn_score_block
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_score_block
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 98,304 bytes (96.0 KB)
//   Total capacity (w/ reuse): 98,304 bytes (96.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   k_block              64x128     f32     32768   [  1,  -1]           -
//   q_block              64x128     f32     32768   [  0,  -1]           -
//   s_block              64x64      f32     16384   [  2,   4]           -
//   s_scaled             64x64      f32     16384   [  4,   5]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void flash_attn_score_block(float* input_q, float* input_k, float* output_s) {
    float q_block[64][128];
    float k_block[64][128];
    float s_block[64][64];
    float s_scaled[64][64];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (2 ops): q_block=TLOAD(input_q,0,0); k_block=TLOAD(input_k,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_q[_row * 128 + _col]);
            vst1q_f32(&q_block[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input_k[_row * 128 + _col]);
            vst1q_f32(&k_block[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            q_block[_row][_col] = input_q[_row * 128 + _col];
            k_block[_row][_col] = input_k[_row * 128 + _col];
        }
    }

    // TMATMUL: s_block = q_block @ k_block
    for (int _i = 0; _i < 64; _i++) {
        for (int _j = 0; _j < 64; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 128; _k++) {
                _sum += q_block[_i][_k] * k_block[_k][_j];}
            s_block[_i][_j] = _sum;}}

    int scale = 0.08838834764831843;

    // FUSED LOOP (2 ops): s_scaled=TMULS(s_block,scalef); output_s=TSTORE(s_scaled,0,0)
    float32x4_t _vs2 = vdupq_n_f32(scalef);
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 64; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&s_block[_row][_col]);
            float32x4_t _vr4 = vmulq_f32(_v3, _vs2);
            vst1q_f32(&s_scaled[_row][_col], _vr4);
            float32x4_t _vs5 = vld1q_f32(&s_scaled[_row][_col]);
            vst1q_f32(&output_s[_row * 64 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 64; _col++) {
            s_scaled[_row][_col] = s_block[_row][_col] * scalef;
            output_s[_row * 64 + _col] = s_scaled[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "flash_attn_score_block"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input_q",
    "input_k",
    "output_s",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(32768),
    (size_t)(32768),
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
    flash_attn_score_block((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER