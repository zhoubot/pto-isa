/**
 * PTO Runtime - Ascend A2/A3 Hardware Intrinsics
 * 
 * This header provides intrinsic implementations for REAL Ascend A2/A3 hardware.
 * It wraps Ascend SDK (CANN) API calls for actual NPU execution.
 * 
 * Requirements:
 * - Ascend SDK (CANN) must be installed
 * - Link with Ascend runtime libraries
 * 
 * Note: This file is only used when compiling for actual hardware.
 * For simulation, see a2a3_intrinsics_sim.h
 */

#ifndef A2A3_INTRINSICS_HW_H
#define A2A3_INTRINSICS_HW_H

// Note: This header is included within a2a3_incore.h which provides:
// - A2A3CoreContext typedef
// - extern "C" wrapping

// =============================================================================
// Hardware-Specific Includes
// =============================================================================

#ifdef A2A3_TARGET_HARDWARE
// Include Ascend SDK headers
// These would be from the actual CANN SDK installation
// #include <acl/acl.h>
// #include <hccl/hccl.h>
// For now, we provide stub definitions

#warning "A2A3 Hardware intrinsics - Ascend SDK headers not available, using stubs"

#endif

// =============================================================================
// Cycle Cost Model (Hardware uses actual execution time)
// =============================================================================

// On real hardware, cycle costs are measured, not estimated
#define A2A3_HW_MEASURE_CYCLES 1

// =============================================================================
// Hardware-Specific Context Extensions
// =============================================================================

typedef struct {
    // Ascend device handle
    void* device_handle;
    
    // Stream for async operations
    void* stream;
    
    // Workspace buffer
    void* workspace;
    int64_t workspace_size;
    
    // Profiling data
    void* profiler;
} A2A3HardwareData;

// =============================================================================
// Hardware Intrinsic Declarations
// =============================================================================

// On real hardware, these map to Ascend SDK calls
// Implementations would be in a2a3_intrinsics_hw.c

#define A2A3_INTRINSIC_IMPL_TYPE "Hardware (CANN SDK)"

#endif // A2A3_INTRINSICS_HW_H
