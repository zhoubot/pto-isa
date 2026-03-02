/**
 * Buffer size limits for runtime address binding (TASSIGN).
 *
 * These macros are intended to be overridden per platform/SoC/toolchain build.
 * When not overridden, we provide conservative defaults derived from common
 * A2/A3/A5 reference environments and existing kernels in this repo.
 */

#ifndef PTO_COMMON_BUFFER_LIMITS_HPP
#define PTO_COMMON_BUFFER_LIMITS_HPP

#include <cstddef>
#include <cstdint>

// NOTE: All sizes/alignments are in BYTES.

// -------------------------------
// UB (Vector core local memory)
// -------------------------------
#ifndef PTO_UBUF_ALIGN_BYTES
#define PTO_UBUF_ALIGN_BYTES (32u)
#endif

#ifndef PTO_UBUF_SIZE_BYTES
#if defined(PTO_NPU_ARCH_A2A3)
#define PTO_UBUF_SIZE_BYTES (192u * 1024u)
#elif defined(PTO_NPU_ARCH_A5)
#define PTO_UBUF_SIZE_BYTES (256u * 1024u)
#else
// Unknown arch: leave a safe-but-small default. Override in build if needed.
#define PTO_UBUF_SIZE_BYTES (192u * 1024u)
#endif
#endif

// -------------------------------
// L1 / CBUF (Cube buffer)
// -------------------------------
#ifndef PTO_CBUF_ALIGN_BYTES
#define PTO_CBUF_ALIGN_BYTES (32u)
#endif

#ifndef PTO_CBUF_SIZE_BYTES
// Many kernels/instr impls assume <= 512KB for static tile shapes in L1.
#define PTO_CBUF_SIZE_BYTES (512u * 1024u)
#endif

// -------------------------------
// L0A / L0B / L0C (Cube buffers)
// -------------------------------
#ifndef PTO_L0A_ALIGN_BYTES
#define PTO_L0A_ALIGN_BYTES (32u)
#endif
#ifndef PTO_L0A_SIZE_BYTES
// Flash-attention kernels in this repo use 0x0 and 0x8000 ping-pong => total 0x10000.
#define PTO_L0A_SIZE_BYTES (0x10000u)
#endif

#ifndef PTO_L0B_ALIGN_BYTES
#define PTO_L0B_ALIGN_BYTES (32u)
#endif
#ifndef PTO_L0B_SIZE_BYTES
#define PTO_L0B_SIZE_BYTES (0x10000u)
#endif

#ifndef PTO_L0C_ALIGN_BYTES
#define PTO_L0C_ALIGN_BYTES (32u)
#endif
#ifndef PTO_L0C_SIZE_BYTES
// Flash-attention kernels use 0x0 and 0x20000 ping-pong => total 0x40000.
#define PTO_L0C_SIZE_BYTES (0x40000u)
#endif

// -------------------------------
// FBUF (Scaling buffer)
// -------------------------------
#ifndef PTO_FBUF_ALIGN_BYTES
#define PTO_FBUF_ALIGN_BYTES (32u)
#endif

#ifndef PTO_FBUF_SIZE_BYTES
// Not consistently specified in current source; must be overridden per platform.
// Provide a placeholder default matching common small on-chip buffers.
#define PTO_FBUF_SIZE_BYTES (64u * 1024u)
#endif

#endif // PTO_COMMON_BUFFER_LIMITS_HPP
