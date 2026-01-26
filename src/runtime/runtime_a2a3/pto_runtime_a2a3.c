/**
 * PTO Runtime System - A2A3 (Ascend) Platform Implementation
 * 
 * This file includes the modular A2A3 implementation from the
 * host/, orchestration/, and core/ subdirectories.
 * 
 * The implementation is split into:
 * - host/a2a3_host.c: Host CPU interface, memory management, workers
 * - orchestration/a2a3_orchestration.c: Task queues, dependency management
 * - core/: InCore intrinsics (header-only, inline implementations)
 */

#include "pto_runtime_a2a3.h"

// Include layer implementations
#include "orchestration/a2a3_orchestration.c"
#include "host/a2a3_host.c"

// Note: Core layer is header-only (intrinsics are inline)
// and included via pto_runtime_a2a3.h -> a2a3_incore.h
