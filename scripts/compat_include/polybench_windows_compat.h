#ifndef CODEX_POLYBENCH_WINDOWS_COMPAT_H
#define CODEX_POLYBENCH_WINDOWS_COMPAT_H

#if defined(_WIN32)
#include <stddef.h>
#include <stdlib.h>

static int codex_posix_memalign(void **memptr, size_t alignment, size_t size)
{
    void *ptr;
    (void)alignment;
    ptr = malloc(size);
    if (!ptr)
        return 12;
    *memptr = ptr;
    return 0;
}

#define posix_memalign codex_posix_memalign
#endif

#endif
