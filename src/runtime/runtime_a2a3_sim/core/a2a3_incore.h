/**
 * PTO Runtime - Ascend A2/A3 Simulator InCore Interface
 * 
 * This is the SIMULATOR version of the InCore interface.
 * It has the same API as the hardware version but uses
 * simulation implementations.
 * 
 * InCore function source code can include this header and
 * will automatically get the correct implementation based
 * on the target platform (hardware vs simulator).
 */

#ifndef A2A3_INCORE_SIM_H
#define A2A3_INCORE_SIM_H

// Force simulator mode
#define A2A3_TARGET_SIMULATOR

// Include the common InCore interface (which will include simulator intrinsics)
#include "../../runtime_a2a3/core/a2a3_incore.h"

#endif // A2A3_INCORE_SIM_H
