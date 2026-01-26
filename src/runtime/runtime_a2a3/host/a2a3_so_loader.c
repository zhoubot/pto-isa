/**
 * PTO Runtime - Ascend A2/A3 Shared Object Loader Implementation
 * 
 * Implements dynamic loading of .so files for orchestration and InCore functions.
 */

#define _GNU_SOURCE  // For dlopen, dlsym, etc.

#include "a2a3_so_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <dirent.h>
#include <sys/stat.h>

// =============================================================================
// Internal State
// =============================================================================

// Orchestration function state
static void* g_orch_handle = NULL;
static A2A3OrchFunc g_orch_func = NULL;

// InCore function registry
static A2A3InCoreFuncEntry g_incore_registry[A2A3_MAX_INCORE_FUNCS];
static int g_incore_count = 0;
static bool g_so_loader_initialized = false;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Extract function name from .so filename.
 * e.g., "tile_add.so" -> "tile_add"
 *       "/path/to/gemm_tile.so" -> "gemm_tile"
 */
static void extract_func_name(const char* so_path, char* func_name, size_t max_len) {
    // Find the last '/' in the path
    const char* filename = strrchr(so_path, '/');
    if (filename) {
        filename++;  // Skip the '/'
    } else {
        filename = so_path;
    }
    
    // Copy filename without extension
    strncpy(func_name, filename, max_len - 1);
    func_name[max_len - 1] = '\0';
    
    // Remove .so extension
    char* dot = strrchr(func_name, '.');
    if (dot && strcmp(dot, ".so") == 0) {
        *dot = '\0';
    }
    
    // Also remove lib prefix if present
    if (strncmp(func_name, "lib", 3) == 0) {
        memmove(func_name, func_name + 3, strlen(func_name + 3) + 1);
    }
}

/**
 * Check if a file has .so extension.
 */
static bool is_so_file(const char* filename) {
    const char* ext = strrchr(filename, '.');
    return ext && strcmp(ext, ".so") == 0;
}

// =============================================================================
// Orchestration Function Loading
// =============================================================================

A2A3OrchFunc a2a3_load_orchestration(const char* so_path, const char* func_name) {
    if (!so_path) {
        fprintf(stderr, "[A2A3 SO Loader] ERROR: so_path is NULL\n");
        return NULL;
    }
    
    // Unload previous orchestration if any
    a2a3_unload_orchestration();
    
    // Load the .so file
    g_orch_handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!g_orch_handle) {
        fprintf(stderr, "[A2A3 SO Loader] ERROR: Failed to load %s: %s\n",
                so_path, dlerror());
        return NULL;
    }
    
    // Determine function name
    const char* name = func_name ? func_name : "orchestration_entry";
    
    // Clear any previous error
    dlerror();
    
    // Load the function
    g_orch_func = (A2A3OrchFunc)dlsym(g_orch_handle, name);
    char* error = dlerror();
    if (error) {
        fprintf(stderr, "[A2A3 SO Loader] ERROR: Failed to find function '%s' in %s: %s\n",
                name, so_path, error);
        dlclose(g_orch_handle);
        g_orch_handle = NULL;
        g_orch_func = NULL;
        return NULL;
    }
    
    printf("[A2A3 SO Loader] Loaded orchestration: %s -> %s()\n", so_path, name);
    return g_orch_func;
}

void a2a3_unload_orchestration(void) {
    if (g_orch_handle) {
        dlclose(g_orch_handle);
        g_orch_handle = NULL;
        g_orch_func = NULL;
    }
}

// =============================================================================
// InCore Function Loading
// =============================================================================

// Include binary loader for .o file support
#include "a2a3_binary_loader.h"

/**
 * Check if a file has .o extension.
 */
static bool is_o_file(const char* filename) {
    const char* ext = strrchr(filename, '.');
    return ext && strcmp(ext, ".o") == 0;
}

int a2a3_load_incore_dir(const char* dir_path, bool is_cube) {
    if (!dir_path) {
        fprintf(stderr, "[A2A3 SO Loader] ERROR: dir_path is NULL\n");
        return A2A3_ERROR_INVALID_CONFIG;
    }
    
    DIR* dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "[A2A3 SO Loader] WARNING: Cannot open directory: %s\n", dir_path);
        return 0;  // Not an error - directory might not exist
    }
    
    int loaded_count = 0;
    int binary_count = 0;
    struct dirent* entry;
    
    // First pass: count .o and .so files
    int o_count = 0, so_count = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (is_o_file(entry->d_name)) o_count++;
        else if (is_so_file(entry->d_name)) so_count++;
    }
    rewinddir(dir);
    
    // Prefer .o files (AICore binaries) over .so files
    bool load_o_files = (o_count > 0);
    
    while ((entry = readdir(dir)) != NULL) {
        // Skip . and ..
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        // Build full path
        char full_path[512];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, entry->d_name);
        
        // Check if it's a regular file
        struct stat st;
        if (stat(full_path, &st) != 0 || !S_ISREG(st.st_mode)) {
            continue;
        }
        
        if (load_o_files) {
            // Load .o files using binary loader (ELF parsing)
            if (is_o_file(entry->d_name)) {
                if (a2a3_load_incore_binary(full_path, NULL, is_cube) == 0) {
                    binary_count++;
                }
            }
        } else {
            // Load .so files using dlopen (for CPU simulation/testing)
            if (is_so_file(entry->d_name)) {
                if (a2a3_load_incore_so(full_path, NULL, is_cube) == 0) {
                    loaded_count++;
                }
            }
        }
    }
    
    closedir(dir);
    
    if (load_o_files) {
        printf("[A2A3 SO Loader] Loaded %d %s binaries (.o) from %s\n",
               binary_count, is_cube ? "AIC" : "AIV", dir_path);
        return binary_count;
    } else {
        printf("[A2A3 SO Loader] Loaded %d %s functions (.so) from %s\n",
               loaded_count, is_cube ? "AIC" : "AIV", dir_path);
        return loaded_count;
    }
}

int a2a3_load_incore_so(const char* so_path, const char* func_name, bool is_cube) {
    if (!so_path) {
        return A2A3_ERROR_INVALID_CONFIG;
    }
    
    if (g_incore_count >= A2A3_MAX_INCORE_FUNCS) {
        fprintf(stderr, "[A2A3 SO Loader] ERROR: Maximum InCore functions reached (%d)\n",
                A2A3_MAX_INCORE_FUNCS);
        return A2A3_ERROR_MEMORY_ALLOC;
    }
    
    // Extract function name if not provided
    char name_buf[128];
    if (!func_name) {
        extract_func_name(so_path, name_buf, sizeof(name_buf));
        func_name = name_buf;
    }
    
    // Check if already loaded
    for (int i = 0; i < g_incore_count; i++) {
        if (g_incore_registry[i].is_loaded &&
            strcmp(g_incore_registry[i].func_name, func_name) == 0) {
            printf("[A2A3 SO Loader] Function '%s' already loaded, skipping\n", func_name);
            return 0;
        }
    }
    
    // Load the .so file
    void* handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "[A2A3 SO Loader] ERROR: Failed to load %s: %s\n",
                so_path, dlerror());
        return A2A3_ERROR_SO_LOAD_FAILED;
    }
    
    // Clear any previous error
    dlerror();
    
    // Try to load the function with the extracted name
    A2A3InCoreFunc func = (A2A3InCoreFunc)dlsym(handle, func_name);
    char* error = dlerror();
    
    // If not found, try with common prefixes/suffixes
    if (error) {
        // Try with "kernel_" prefix
        char alt_name[128];
        snprintf(alt_name, sizeof(alt_name), "kernel_%s", func_name);
        func = (A2A3InCoreFunc)dlsym(handle, alt_name);
        error = dlerror();
        
        if (error) {
            // Try with "_entry" suffix
            snprintf(alt_name, sizeof(alt_name), "%s_entry", func_name);
            func = (A2A3InCoreFunc)dlsym(handle, alt_name);
            error = dlerror();
        }
    }
    
    if (error || !func) {
        fprintf(stderr, "[A2A3 SO Loader] WARNING: No function found in %s\n", so_path);
        dlclose(handle);
        return A2A3_ERROR_FUNC_NOT_FOUND;
    }
    
    // Register the function
    A2A3InCoreFuncEntry* entry = &g_incore_registry[g_incore_count];
    strncpy(entry->func_name, func_name, sizeof(entry->func_name) - 1);
    entry->func_name[sizeof(entry->func_name) - 1] = '\0';
    entry->func_ptr = func;
    entry->so_handle = handle;
    entry->is_cube = is_cube;
    entry->is_loaded = true;
    g_incore_count++;
    
    printf("[A2A3 SO Loader] Loaded InCore: %s -> %s() [%s]\n",
           so_path, func_name, is_cube ? "AIC" : "AIV");
    
    return 0;
}

A2A3InCoreFunc a2a3_lookup_incore(const char* func_name) {
    if (!func_name) return NULL;
    
    for (int i = 0; i < g_incore_count; i++) {
        if (g_incore_registry[i].is_loaded &&
            strcmp(g_incore_registry[i].func_name, func_name) == 0) {
            return g_incore_registry[i].func_ptr;
        }
    }
    
    return NULL;
}

bool a2a3_is_cube_func(const char* func_name) {
    if (!func_name) return false;
    
    for (int i = 0; i < g_incore_count; i++) {
        if (g_incore_registry[i].is_loaded &&
            strcmp(g_incore_registry[i].func_name, func_name) == 0) {
            return g_incore_registry[i].is_cube;
        }
    }
    
    return false;
}

int a2a3_register_incore(const char* func_name, A2A3InCoreFunc func_ptr, bool is_cube) {
    if (!func_name || !func_ptr) {
        return A2A3_ERROR_INVALID_CONFIG;
    }
    
    if (g_incore_count >= A2A3_MAX_INCORE_FUNCS) {
        return A2A3_ERROR_MEMORY_ALLOC;
    }
    
    // Check if already registered
    for (int i = 0; i < g_incore_count; i++) {
        if (g_incore_registry[i].is_loaded &&
            strcmp(g_incore_registry[i].func_name, func_name) == 0) {
            // Update existing entry
            g_incore_registry[i].func_ptr = func_ptr;
            g_incore_registry[i].is_cube = is_cube;
            return 0;
        }
    }
    
    // Add new entry
    A2A3InCoreFuncEntry* entry = &g_incore_registry[g_incore_count];
    strncpy(entry->func_name, func_name, sizeof(entry->func_name) - 1);
    entry->func_name[sizeof(entry->func_name) - 1] = '\0';
    entry->func_ptr = func_ptr;
    entry->so_handle = NULL;  // Not loaded from .so
    entry->is_cube = is_cube;
    entry->is_loaded = true;
    g_incore_count++;
    
    return 0;
}

int a2a3_get_incore_count(void) {
    // Return total count: .so functions + .o binaries
    return g_incore_count + a2a3_get_incore_binary_count();
}

void a2a3_unload_all_incore(void) {
    for (int i = 0; i < g_incore_count; i++) {
        if (g_incore_registry[i].is_loaded && g_incore_registry[i].so_handle) {
            dlclose(g_incore_registry[i].so_handle);
        }
        g_incore_registry[i].is_loaded = false;
        g_incore_registry[i].so_handle = NULL;
        g_incore_registry[i].func_ptr = NULL;
    }
    g_incore_count = 0;
}

// =============================================================================
// Initialization and Cleanup
// =============================================================================

void a2a3_so_loader_init(void) {
    if (g_so_loader_initialized) return;
    
    // Clear registry
    memset(g_incore_registry, 0, sizeof(g_incore_registry));
    g_incore_count = 0;
    g_orch_handle = NULL;
    g_orch_func = NULL;
    g_so_loader_initialized = true;
    
    // Initialize binary loader for .o file support
    a2a3_binary_loader_init();
    
    printf("[A2A3 SO Loader] Initialized\n");
}

void a2a3_so_loader_cleanup(void) {
    if (!g_so_loader_initialized) return;
    
    a2a3_unload_orchestration();
    a2a3_unload_all_incore();
    
    // Cleanup binary loader
    a2a3_binary_loader_cleanup();
    
    g_so_loader_initialized = false;
    
    printf("[A2A3 SO Loader] Cleanup complete\n");
}
