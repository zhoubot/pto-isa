#include <dlfcn.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <algorithm>
#include <string>
#include <vector>

using pto_program_name_fn = const char *(*)();
using pto_num_memrefs_fn = int (*)();
using pto_memref_name_fn = const char *(*)(int);
using pto_memref_bytes_fn = size_t (*)(int);
using pto_memref_dtype_fn = const char *(*)(int);
using pto_memref_elem_bytes_fn = size_t (*)(int);
using pto_memref_is_output_fn = int (*)(int);
using pto_launch_fn = void (*)(void **, void *);

static uint64_t checksum_bytes(const void *data, size_t bytes) {
    const uint8_t *p = static_cast<const uint8_t *>(data);
    uint64_t h = 1469598103934665603ull; // FNV-1a
    for (size_t i = 0; i < bytes; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static void fill_bytes(void *data, size_t bytes, uint32_t seed) {
    uint8_t *p = static_cast<uint8_t *>(data);
    uint32_t x = seed ? seed : 1u;
    for (size_t i = 0; i < bytes; i++) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        p[i] = static_cast<uint8_t>(x & 0xff);
    }
}

static void fill_by_dtype(void *data, size_t bytes, const char *dtype, uint32_t seed) {
    if (!dtype || dtype[0] == '\0') {
        fill_bytes(data, bytes, seed);
        return;
    }
    if (strcmp(dtype, "f32") == 0) {
        float *p = static_cast<float *>(data);
        const size_t n = bytes / sizeof(float);
        for (size_t i = 0; i < n; i++) {
            p[i] = 0.1f * (1.0f + (float)((seed + i) % 97));
        }
        return;
    }
    if (strcmp(dtype, "f64") == 0) {
        double *p = static_cast<double *>(data);
        const size_t n = bytes / sizeof(double);
        for (size_t i = 0; i < n; i++) {
            p[i] = 0.1 * (1.0 + (double)((seed + i) % 97));
        }
        return;
    }
    if (strcmp(dtype, "f16") == 0 || strcmp(dtype, "bf16") == 0) {
        uint16_t *p = static_cast<uint16_t *>(data);
        const size_t n = bytes / sizeof(uint16_t);
        const uint16_t val = (strcmp(dtype, "f16") == 0) ? 0x3c00 : 0x3f80; // 1.0
        for (size_t i = 0; i < n; i++) {
            p[i] = val;
        }
        return;
    }
    fill_bytes(data, bytes, seed);
}

static bool is_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return false;
    }
    return S_ISDIR(st.st_mode);
}

static bool has_suffix(const std::string &s, const char *suffix) {
    const size_t n = strlen(suffix);
    if (s.size() < n) {
        return false;
    }
    return s.compare(s.size() - n, n, suffix) == 0;
}

static void collect_shared_objects(const char *path, std::vector<std::string> &out) {
    if (!is_dir(path)) {
        out.emplace_back(path);
        return;
    }
    DIR *dir = opendir(path);
    if (!dir) {
        return;
    }
    while (dirent *ent = readdir(dir)) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }
        std::string full = std::string(path) + "/" + ent->d_name;
        if (is_dir(full.c_str())) {
            collect_shared_objects(full.c_str(), out);
            continue;
        }
        if (has_suffix(full, ".so")) {
            out.emplace_back(std::move(full));
        }
    }
    closedir(dir);
}

template <typename T>
static T load_symbol(void *handle, const char *name, const char *so_path) {
    void *sym = dlsym(handle, name);
    if (!sym) {
        fprintf(stderr, "[skip] %s: missing symbol %s\n", so_path, name);
        return nullptr;
    }
    return reinterpret_cast<T>(sym);
}

template <typename T>
static T load_optional_symbol(void *handle, const char *name) {
    void *sym = dlsym(handle, name);
    return reinterpret_cast<T>(sym);
}

static int run_one_so(const std::string &so_path) {
    void *handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "[fail] dlopen %s: %s\n", so_path.c_str(), dlerror());
        return 1;
    }

    auto pto_program_name = load_symbol<pto_program_name_fn>(handle, "pto_program_name", so_path.c_str());
    auto pto_num_memrefs = load_symbol<pto_num_memrefs_fn>(handle, "pto_num_memrefs", so_path.c_str());
    auto pto_memref_name = load_symbol<pto_memref_name_fn>(handle, "pto_memref_name", so_path.c_str());
    auto pto_memref_bytes = load_symbol<pto_memref_bytes_fn>(handle, "pto_memref_bytes", so_path.c_str());
    auto pto_memref_dtype = load_optional_symbol<pto_memref_dtype_fn>(handle, "pto_memref_dtype");
    auto pto_memref_elem_bytes = load_optional_symbol<pto_memref_elem_bytes_fn>(handle, "pto_memref_elem_bytes");
    auto pto_memref_is_output = load_symbol<pto_memref_is_output_fn>(handle, "pto_memref_is_output", so_path.c_str());
    auto pto_launch = load_symbol<pto_launch_fn>(handle, "pto_launch", so_path.c_str());
    if (!pto_program_name || !pto_num_memrefs || !pto_memref_name || !pto_memref_bytes || !pto_memref_is_output || !pto_launch) {
        dlclose(handle);
        return 2;
    }

    const int n = pto_num_memrefs();
    if (n <= 0) {
        fprintf(stderr, "[skip] %s: no memrefs\n", so_path.c_str());
        dlclose(handle);
        return 0;
    }

    std::vector<void *> host_ptrs(n, nullptr);
    std::vector<size_t> bytes(n, 0);

    int rc = 0;
    for (int i = 0; i < n; i++) {
        bytes[i] = pto_memref_bytes(i);
        if (bytes[i] == 0) {
            bytes[i] = 1;
        }
        host_ptrs[i] = malloc(bytes[i]);
        if (!host_ptrs[i]) {
            fprintf(stderr, "[fail] %s: malloc(%zu) idx=%d\n", so_path.c_str(), bytes[i], i);
            rc = 3;
            break;
        }

        if (pto_memref_is_output(i)) {
            memset(host_ptrs[i], 0, bytes[i]);
        } else {
            const char *dtype = pto_memref_dtype ? pto_memref_dtype(i) : nullptr;
            (void)pto_memref_elem_bytes;
            fill_by_dtype(host_ptrs[i], bytes[i], dtype, (uint32_t)(i + 1));
        }
    }

    if (rc == 0) {
        printf("[run] %s (%s) memrefs=%d\n", so_path.c_str(), pto_program_name(), n);
        fflush(stdout);
        pto_launch(host_ptrs.data(), nullptr);
    }

    if (rc == 0) {
        for (int i = 0; i < n; i++) {
            if (!pto_memref_is_output(i)) {
                continue;
            }
            const uint64_t h = checksum_bytes(host_ptrs[i], bytes[i]);
            printf("  [out] %s bytes=%zu checksum=0x%016llx\n", pto_memref_name(i), bytes[i], (unsigned long long)h);
            fflush(stdout);
        }
    }

    for (int i = 0; i < n; i++) {
        if (host_ptrs[i]) {
            free(host_ptrs[i]);
        }
    }

    dlclose(handle);
    return rc;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <kernel.so|dir> [more...]\n", argv[0]);
        return 2;
    }

    setvbuf(stdout, nullptr, _IOLBF, 0);

    std::vector<std::string> so_paths;
    for (int i = 1; i < argc; i++) {
        collect_shared_objects(argv[i], so_paths);
    }
    std::sort(so_paths.begin(), so_paths.end());
    so_paths.erase(std::unique(so_paths.begin(), so_paths.end()), so_paths.end());

    if (so_paths.empty()) {
        fprintf(stderr, "No .so files found.\n");
        return 2;
    }

    int failures = 0;
    for (const auto &p : so_paths) {
        int rc = run_one_so(p);
        if (rc != 0) {
            failures++;
        }
    }

    if (failures != 0) {
        fprintf(stderr, "Failures: %d\n", failures);
        return 1;
    }
    return 0;
}
