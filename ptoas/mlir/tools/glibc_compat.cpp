#include <errno.h>
#include <limits.h>
#include <stddef.h>

namespace {

inline bool is_space(unsigned char c) {
    switch (c) {
        case ' ':
        case '\t':
        case '\n':
        case '\v':
        case '\f':
        case '\r':
            return true;
        default:
            return false;
    }
}

inline int digit_value(unsigned char c) {
    if (c >= '0' && c <= '9') {
        return static_cast<int>(c - '0');
    }
    if (c >= 'a' && c <= 'z') {
        return static_cast<int>(c - 'a') + 10;
    }
    if (c >= 'A' && c <= 'Z') {
        return static_cast<int>(c - 'A') + 10;
    }
    return -1;
}

struct ParsedUll {
    unsigned long long value;
    const char* end;
    bool any;
    bool negative;
    bool overflow;
    bool invalid_base;
};

ParsedUll parse_ull(const char* nptr, int base) {
    const char* s = nptr;
    while (*s && is_space(static_cast<unsigned char>(*s))) {
        ++s;
    }

    bool negative = false;
    if (*s == '+' || *s == '-') {
        negative = (*s == '-');
        ++s;
    }

    if (base != 0 && (base < 2 || base > 36)) {
        return {0ULL, nptr, false, negative, false, true};
    }

    int b = base;
    const char* digits = s;

    // Base/prefix handling:
    // - ISO C2x/C23 adds support for 0b/0B with base 0 or 2.
    // - 0x/0X is supported with base 0 or 16.
    if (b == 0) {
        if (digits[0] == '0') {
            if (digits[1] == 'x' || digits[1] == 'X') {
                b = 16;
                digits += 2;
            } else if (digits[1] == 'b' || digits[1] == 'B') {
                b = 2;
                digits += 2;
            } else {
                b = 8;
                // Keep the leading '0' as a digit (no prefix to skip).
            }
        } else {
            b = 10;
        }
    } else if (b == 16) {
        if (digits[0] == '0' && (digits[1] == 'x' || digits[1] == 'X')) {
            digits += 2;
        }
    } else if (b == 2) {
        if (digits[0] == '0' && (digits[1] == 'b' || digits[1] == 'B')) {
            digits += 2;
        }
    }

    unsigned long long acc = 0;
    bool any = false;
    bool overflow = false;
    const char* p = digits;
    const unsigned ub = static_cast<unsigned>(b);

    for (; *p; ++p) {
        int d = digit_value(static_cast<unsigned char>(*p));
        if (d < 0 || static_cast<unsigned>(d) >= ub) {
            break;
        }
        any = true;
        if (overflow) {
            continue;
        }
        unsigned long long ud = static_cast<unsigned long long>(d);
        unsigned long long limit = ULLONG_MAX / ub;
        unsigned long long rem = ULLONG_MAX % ub;
        if (acc > limit || (acc == limit && ud > rem)) {
            overflow = true;
            acc = ULLONG_MAX;
            continue;
        }
        acc = acc * ub + ud;
    }

    // If no digits were consumed, no conversion happened.
    if (!any) {
        return {0ULL, nptr, false, negative, false, false};
    }
    return {acc, p, true, negative, overflow, false};
}

}  // namespace

// glibc 2.38 introduced __isoc23_strto* symbols and conditionally redirects
// strto* in headers when ISO C2x extensions are enabled (e.g. via _GNU_SOURCE).
// LLVM/MLIR (often built with -D_GNU_SOURCE) may therefore end up requiring
// GLIBC_2.38 at runtime.
//
// To keep `ptoas` runnable on glibc 2.36, we provide local definitions for the
// subset of __isoc23_strto* that LLVM commonly references. These avoid calling
// libc's strto* to prevent recursion on glibc 2.38, and implement the C23 0b/0B
// prefix behavior for base 0/2.

extern "C" unsigned long long __isoc23_strtoull(const char* nptr, char** endptr, int base) {
    ParsedUll parsed = parse_ull(nptr, base);
    if (parsed.invalid_base) {
        errno = EINVAL;
        if (endptr) {
            *endptr = const_cast<char*>(nptr);
        }
        return 0ULL;
    }
    if (endptr) {
        *endptr = const_cast<char*>(parsed.end);
    }
    if (!parsed.any) {
        return 0ULL;
    }
    if (parsed.overflow) {
        errno = ERANGE;
        return ULLONG_MAX;
    }
    if (parsed.negative) {
        return 0ULL - parsed.value;
    }
    return parsed.value;
}

extern "C" long long __isoc23_strtoll(const char* nptr, char** endptr, int base) {
    ParsedUll parsed = parse_ull(nptr, base);
    if (parsed.invalid_base) {
        errno = EINVAL;
        if (endptr) {
            *endptr = const_cast<char*>(nptr);
        }
        return 0LL;
    }
    if (endptr) {
        *endptr = const_cast<char*>(parsed.end);
    }
    if (!parsed.any) {
        return 0LL;
    }

    // Clamp on overflow.
    if (parsed.negative) {
        const unsigned long long neg_limit = static_cast<unsigned long long>(LLONG_MAX) + 1ULL;
        if (parsed.overflow || parsed.value > neg_limit) {
            errno = ERANGE;
            return LLONG_MIN;
        }
        if (parsed.value == neg_limit) {
            return LLONG_MIN;
        }
        return -static_cast<long long>(parsed.value);
    }

    if (parsed.overflow || parsed.value > static_cast<unsigned long long>(LLONG_MAX)) {
        errno = ERANGE;
        return LLONG_MAX;
    }
    return static_cast<long long>(parsed.value);
}

extern "C" long int __isoc23_strtol(const char* nptr, char** endptr, int base) {
    ParsedUll parsed = parse_ull(nptr, base);
    if (parsed.invalid_base) {
        errno = EINVAL;
        if (endptr) {
            *endptr = const_cast<char*>(nptr);
        }
        return 0L;
    }
    if (endptr) {
        *endptr = const_cast<char*>(parsed.end);
    }
    if (!parsed.any) {
        return 0L;
    }

    // Clamp on overflow.
    if (parsed.negative) {
        const unsigned long long neg_limit = static_cast<unsigned long long>(LONG_MAX) + 1ULL;
        if (parsed.overflow || parsed.value > neg_limit) {
            errno = ERANGE;
            return LONG_MIN;
        }
        if (parsed.value == neg_limit) {
            return LONG_MIN;
        }
        return -static_cast<long int>(parsed.value);
    }

    if (parsed.overflow || parsed.value > static_cast<unsigned long long>(LONG_MAX)) {
        errno = ERANGE;
        return LONG_MAX;
    }
    return static_cast<long int>(parsed.value);
}

