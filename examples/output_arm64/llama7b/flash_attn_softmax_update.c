// PTO Program: flash_attn_softmax_update
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_softmax_update
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     12
//   Total capacity (no reuse): 51,456 bytes (50.2 KB)
//   Total capacity (w/ reuse): 34,048 bytes (33.2 KB)
//   Reuse savings:            17,408 bytes (33.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   l_new                64x1       f32       256   [ 11,  13]           -
//   l_prev               64x1       f32       256   [  2,   9]           -
//   l_scaled             64x1       f32       256   [  9,  11]           <- m_diff
//   m_cur                64x1       f32       256   [  3,   4]           -
//   m_diff               64x1       f32       256   [  7,   8]           <- m_cur
//   m_new                64x1       f32       256   [  4,  12]           -
//   m_prev               64x1       f32       256   [  1,   7]           -
//   p_block              64x64      f32     16384   [  6,  14]           <- s_block
//   p_rowsum             64x1       f32       256   [ 10,  11]           <- l_prev
//   s_block              64x64      f32     16384   [  0,   5]           -
//   s_shifted            64x64      f32     16384   [  5,   6]           -
//   scale_old            64x1       f32       256   [  8,  15]           <- m_prev
//
// BUFFER REUSE MAP:
//   p_block reuses buffer of s_block
//   scale_old reuses buffer of m_prev
//   m_diff reuses buffer of m_cur
//   l_scaled reuses buffer of m_diff
//   p_rowsum reuses buffer of l_prev
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void flash_attn_softmax_update(float* input_s, float* input_m_prev, float* input_l_prev, float* output_m_new, float* output_l_new, float* output_p, float* output_scale_old) {
    float s_block[64][64];
    float m_prev[64][1];
    float l_prev[64][1];
    float m_new[64][1];
    float m_cur[64][1];
    float l_new[64][1];
    float p_block[64][64];
    float s_shifted[64][64];
    float scale_old[64][1];
    float m_diff[64][1];
    float l_scaled[64][1];
    float p_rowsum[64][1];

    // Loop fusion: 6 loop overheads saved

    // FUSED LOOP (1 ops): s_block=TLOAD(input_s,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 64; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_s[_row * 64 + _col]);
            vst1q_f32(&s_block[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 64; _col++) {
            s_block[_row][_col] = input_s[_row * 64 + _col];
        }
    }

    // FUSED LOOP (2 ops): m_prev=TLOAD(input_m_prev,0,0); l_prev=TLOAD(input_l_prev,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_m_prev[_row * 1 + _col]);
            vst1q_f32(&m_prev[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&input_l_prev[_row * 1 + _col]);
            vst1q_f32(&l_prev[_row][_col], _vl2);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            m_prev[_row][_col] = input_m_prev[_row * 1 + _col];
            l_prev[_row][_col] = input_l_prev[_row * 1 + _col];
        }
    }

    // TROWMAX: m_cur = rowmax(s_block)
    for (int _row = 0; _row < 64; _row++) {
        float _max = s_block[_row][0];
        for (int _col = 1; _col < 64; _col++) {
            if (s_block[_row][_col] > _max) _max = s_block[_row][_col];
        }
        m_cur[_row][0] = _max;}

    // FUSED LOOP (1 ops): m_new=TMAX(m_prev,m_cur)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&m_prev[_row][_col]);
            float32x4_t _v4 = vld1q_f32(&m_cur[_row][_col]);
            float32x4_t _vr5 = vmaxq_f32(_v3, _v4);
            vst1q_f32(&m_new[_row][_col], _vr5);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            m_new[_row][_col] = m_prev[_row][_col] + m_cur[_row][_col];
        }
    }

    // FUSED LOOP (2 ops): s_shifted=TROWEXPANDSUB(s_block,m_new); p_block=TEXP(s_shifted)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 64; _col += 4) {
            float32x4_t _v06 = vld1q_f32(&s_block[_row][_col]);
            float32x4_t _vb8 = vdupq_n_f32(m_new[_row][0]);
            float32x4_t _vr7 = vsubq_f32(_v06, _vb8);
            vst1q_f32(&s_shifted[_row][_col], _vr7);
            float32x4_t _v9 = vld1q_f32(&s_shifted[_row][_col]);
            float32x4_t _vr10 = _v9;
            vst1q_f32(&p_block[_row][_col], _vr10);
        }
        // Scalar cleanup
        for (; _col < 64; _col++) {
            s_shifted[_row][_col] = s_block[_row][_col] - m_new[_row][0];
            p_block[_row][_col] = expf(s_shifted[_row][_col]);
        }
    }

    // FUSED LOOP (3 ops): m_diff=TSUB(m_prev,m_new); scale_old=TEXP(m_diff); l_scaled=TMUL(scale_old,l_prev)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v11 = vld1q_f32(&m_prev[_row][_col]);
            float32x4_t _v12 = vld1q_f32(&m_new[_row][_col]);
            float32x4_t _vr13 = vsubq_f32(_v11, _v12);
            vst1q_f32(&m_diff[_row][_col], _vr13);
            float32x4_t _v14 = vld1q_f32(&m_diff[_row][_col]);
            float32x4_t _vr15 = _v14;
            vst1q_f32(&scale_old[_row][_col], _vr15);
            float32x4_t _v16 = vld1q_f32(&scale_old[_row][_col]);
            float32x4_t _v17 = vld1q_f32(&l_prev[_row][_col]);
            float32x4_t _vr18 = vmulq_f32(_v16, _v17);
            vst1q_f32(&l_scaled[_row][_col], _vr18);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            m_diff[_row][_col] = m_prev[_row][_col] - m_new[_row][_col];
            scale_old[_row][_col] = expf(m_diff[_row][_col]);
            l_scaled[_row][_col] = scale_old[_row][_col] * l_prev[_row][_col];
        }
    }

    // TROWSUM: p_rowsum = rowsum(p_block)
    for (int _row = 0; _row < 64; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 64; _col++) {
            _sum += p_block[_row][_col];
        }
        p_rowsum[_row][0] = _sum;}

    // FUSED LOOP (3 ops): l_new=TADD(l_scaled,p_rowsum); output_m_new=TSTORE(m_new,0,0); output_l_new=TSTORE(l_new,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v19 = vld1q_f32(&l_scaled[_row][_col]);
            float32x4_t _v20 = vld1q_f32(&p_rowsum[_row][_col]);
            float32x4_t _vr21 = vaddq_f32(_v19, _v20);
            vst1q_f32(&l_new[_row][_col], _vr21);
            float32x4_t _vs22 = vld1q_f32(&m_new[_row][_col]);
            vst1q_f32(&output_m_new[_row * 1 + _col], _vs22);
            float32x4_t _vs23 = vld1q_f32(&l_new[_row][_col]);
            vst1q_f32(&output_l_new[_row * 1 + _col], _vs23);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            l_new[_row][_col] = l_scaled[_row][_col] + p_rowsum[_row][_col];
            output_m_new[_row * 1 + _col] = m_new[_row][_col];
            output_l_new[_row * 1 + _col] = l_new[_row][_col];
        }
    }

    // FUSED LOOP (1 ops): output_p=TSTORE(p_block,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 64; _col += 4) {
            float32x4_t _vs24 = vld1q_f32(&p_block[_row][_col]);
            vst1q_f32(&output_p[_row * 64 + _col], _vs24);
        }
        // Scalar cleanup
        for (; _col < 64; _col++) {
            output_p[_row * 64 + _col] = p_block[_row][_col];
        }
    }

    // FUSED LOOP (1 ops): output_scale_old=TSTORE(scale_old,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vs25 = vld1q_f32(&scale_old[_row][_col]);
            vst1q_f32(&output_scale_old[_row * 1 + _col], _vs25);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            output_scale_old[_row * 1 + _col] = scale_old[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "flash_attn_softmax_update"; }
enum { kPtoNumMemrefs = 7 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input_s",
    "input_m_prev",
    "input_l_prev",
    "output_m_new",
    "output_l_new",
    "output_p",
    "output_scale_old",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(16384),
    (size_t)(256),
    (size_t)(256),
    (size_t)(256),
    (size_t)(256),
    (size_t)(16384),
    (size_t)(256),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    0,
    0,
    1,
    1,
    1,
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
    flash_attn_softmax_update((float*)args[0], (float*)args[1], (float*)args[2], (float*)args[3], (float*)args[4], (float*)args[5], (float*)args[6]);
}
#endif  // PTO_CPU_SMOKE_RUNNER