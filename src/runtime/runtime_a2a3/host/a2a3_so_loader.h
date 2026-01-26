/**
 * PTO Runtime - Ascend A2/A3 Shared Object Loader
 * 
 * This module handles dynamic loading of .so files for:
 * - Orchestration functions
 * - InCore AIV functions
 * - InCore AIC functions
 */

#ifndef A2A3_SO_LOADER_H
#define A2A3_SO_LOADER_H

#include "../a2a3_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Orchestration Function Loading
// =============================================================================

/**
 * Orchestration function signature.
 * The orchestration function builds the task graph using runtime APIs.
 */
typedef void (*A2A3OrchFunc)(void* runtime, void* user_data);

/**
 * Load orchestration function from .so file.
 * 
 * The .so file should export a function named "orchestration_entry"
 * or the function name can be specified.
 * 
 * @param so_path    Path to the .so file
 * @param func_name  Function name to load (NULL for default "orchestration_entry")
 * @return Function pointer, or NULL on failure
 */
A2A3OrchFunc a2a3_load_orchestration(const char* so_path, const char* func_name);

/**
 * Unload orchestration .so file.
 */
void a2a3_unload_orchestration(void);

// =============================================================================
// InCore Function Loading
// =============================================================================

/**
 * InCore function entry in the registry.
 */
typedef struct {
    char func_name[128];       // Function name
    A2A3InCoreFunc func_ptr;   // Function pointer
    void* so_handle;           // dlopen handle (for unloading)
    bool is_cube;              // True if AIC (Cube), false if AIV (Vector)
    bool is_loaded;            // Entry is valid
} A2A3InCoreFuncEntry;

/**
 * Load all InCore functions from a directory.
 * 
 * Scans the directory for .so files and loads each one.
 * Each .so should export a function with the same name as the file
 * (e.g., "tile_add.so" exports "tile_add").
 * 
 * @param dir_path  Path to directory containing .so files
 * @param is_cube   True if loading AIC (Cube) functions
 * @return Number of functions loaded, or negative error code
 */
int a2a3_load_incore_dir(const char* dir_path, bool is_cube);

/**
 * Load a single InCore function from .so file.
 * 
 * @param so_path   Path to the .so file
 * @param func_name Function name (NULL to derive from filename)
 * @param is_cube   True if AIC (Cube) function
 * @return 0 on success, negative error code on failure
 */
int a2a3_load_incore_so(const char* so_path, const char* func_name, bool is_cube);

/**
 * Lookup InCore function by name.
 * 
 * @param func_name  Function name
 * @return Function pointer, or NULL if not found
 */
A2A3InCoreFunc a2a3_lookup_incore(const char* func_name);

/**
 * Check if a function is a Cube function.
 * 
 * @param func_name  Function name
 * @return true if Cube, false if Vector or not found
 */
bool a2a3_is_cube_func(const char* func_name);

/**
 * Register an InCore function manually (without loading from .so).
 * 
 * @param func_name  Function name
 * @param func_ptr   Function pointer
 * @param is_cube    True if AIC (Cube) function
 * @return 0 on success, negative error code on failure
 */
int a2a3_register_incore(const char* func_name, A2A3InCoreFunc func_ptr, bool is_cube);

/**
 * Get number of loaded InCore functions.
 */
int a2a3_get_incore_count(void);

/**
 * Unload all InCore functions.
 */
void a2a3_unload_all_incore(void);

// =============================================================================
// Initialization and Cleanup
// =============================================================================

/**
 * Initialize the SO loader module.
 */
void a2a3_so_loader_init(void);

/**
 * Cleanup the SO loader module and unload all .so files.
 */
void a2a3_so_loader_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // A2A3_SO_LOADER_H
