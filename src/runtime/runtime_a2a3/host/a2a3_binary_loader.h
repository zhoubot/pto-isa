/**
 * PTO Runtime - Ascend A2/A3 Binary Loader
 * 
 * This module provides utilities for loading AICore kernel binaries from
 * compiled .o files (ELF format) and extracting the executable .text section.
 *
 * Process:
 * 1. Read .o file (ELF format)
 * 2. Parse ELF header
 * 3. Extract .text section (executable code)
 * 4. Copy to device GM memory
 * 5. Return device address for function pointer dispatch
 *
 * Based on ref_runtime/src/platform/a2a3/host/binary_loader.cpp
 */

#ifndef A2A3_BINARY_LOADER_H
#define A2A3_BINARY_LOADER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// InCore Binary Entry
// =============================================================================

/**
 * InCore binary entry for ELF .o files.
 * Stores the loaded binary data and device address.
 */
typedef struct {
    char func_name[128];       // Function name
    uint8_t* binary_data;      // Extracted .text section binary
    size_t binary_size;        // Size of binary data
    uint64_t device_addr;      // Device GM address (for real hardware)
    bool is_cube;              // True if AIC (Cube), false if AIV (Vector)
    bool is_loaded;            // Entry is valid
} A2A3InCoreBinaryEntry;

// =============================================================================
// ELF Binary Loading Functions
// =============================================================================

/**
 * Get file size in bytes.
 *
 * @param file_path  Path to the file
 * @return File size in bytes, or 0 on error
 */
uint32_t a2a3_get_file_size(const char* file_path);

/**
 * Load and extract .text section from ELF .o file.
 *
 * This function:
 * 1. Reads the entire .o file into memory
 * 2. Parses the ELF64 header
 * 3. Locates the .text section (executable code)
 * 4. Extracts and returns the .text section binary data
 *
 * @param bin_path     Path to the .o file (ELF format)
 * @param out_data     Output: pointer to allocated binary data (caller must free)
 * @param out_size     Output: size of binary data
 * @return 0 on success, negative error code on failure
 */
int a2a3_load_elf_text_section(const char* bin_path, uint8_t** out_data, size_t* out_size);

/**
 * Load all InCore binaries from a directory.
 * 
 * Scans the directory for .o files and loads each one.
 * 
 * @param dir_path  Path to directory containing .o files
 * @param is_cube   True if loading AIC (Cube) functions
 * @return Number of binaries loaded, or negative error code
 */
int a2a3_load_incore_binary_dir(const char* dir_path, bool is_cube);

/**
 * Load a single InCore binary from .o file.
 * 
 * @param bin_path   Path to the .o file
 * @param func_name  Function name (NULL to derive from filename)
 * @param is_cube    True if AIC (Cube) function
 * @return 0 on success, negative error code on failure
 */
int a2a3_load_incore_binary(const char* bin_path, const char* func_name, bool is_cube);

/**
 * Lookup InCore binary by name.
 * 
 * @param func_name  Function name
 * @return Pointer to binary entry, or NULL if not found
 */
A2A3InCoreBinaryEntry* a2a3_lookup_incore_binary(const char* func_name);

/**
 * Get number of loaded InCore binaries.
 */
int a2a3_get_incore_binary_count(void);

/**
 * Unload all InCore binaries and free memory.
 */
void a2a3_unload_all_incore_binaries(void);

// =============================================================================
// Initialization and Cleanup
// =============================================================================

/**
 * Initialize the binary loader module.
 */
void a2a3_binary_loader_init(void);

/**
 * Cleanup the binary loader module.
 */
void a2a3_binary_loader_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // A2A3_BINARY_LOADER_H
