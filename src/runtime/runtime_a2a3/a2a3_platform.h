/**
 * PTO Runtime - Ascend A2/A3 Platform Unified Header
 * 
 * This header provides a unified interface to the A2A3 platform,
 * including all three layers:
 * - Host: CPU-side control and memory management
 * - Orchestration: Task scheduling and dependency management
 * - Core: InCore function execution on AI cores
 * 
 * Include this header to get access to the complete A2A3 API.
 */

#ifndef A2A3_PLATFORM_H
#define A2A3_PLATFORM_H

// =============================================================================
// Layer Headers
// =============================================================================

#include "host/a2a3_host.h"
#include "orchestration/a2a3_orchestration.h"
#include "core/a2a3_incore.h"

// =============================================================================
// Platform Configuration
// =============================================================================

// Ascend 910B (A2/A3) Architecture
#define A2A3_PLATFORM_NAME          "Ascend 910B"
#define A2A3_PLATFORM_GENERATION    "A2/A3"

// Core configuration
#define A2A3_VECTOR_CORES           48
#define A2A3_CUBE_CORES             24
#define A2A3_TOTAL_CORES            (A2A3_VECTOR_CORES + A2A3_CUBE_CORES)

// Memory hierarchy
#define A2A3_GM_SIZE_GB             32      // Global Memory (DDR)
#define A2A3_L2_SIZE_MB             200     // L2 Cache (shared)
#define A2A3_L1_SIZE_KB             192     // L1/UB per core

// Clock frequency (approximate)
#define A2A3_CLOCK_MHZ              1800    // 1.8 GHz

// Compute capability
#define A2A3_FP16_TFLOPS            320
#define A2A3_FP32_TFLOPS            160

// =============================================================================
// Convenience Macros
// =============================================================================

// Check if running on simulator
#ifdef A2A3_TARGET_SIMULATOR
    #define A2A3_IS_SIMULATOR  1
#else
    #define A2A3_IS_SIMULATOR  0
#endif

// Memory size in bytes
#define A2A3_GM_SIZE_BYTES   ((int64_t)A2A3_GM_SIZE_GB * 1024 * 1024 * 1024)
#define A2A3_L2_SIZE_BYTES   ((int64_t)A2A3_L2_SIZE_MB * 1024 * 1024)
#define A2A3_L1_SIZE_BYTES   ((int64_t)A2A3_L1_SIZE_KB * 1024)

#endif // A2A3_PLATFORM_H
