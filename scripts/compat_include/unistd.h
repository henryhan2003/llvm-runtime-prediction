#ifndef CODEX_COMPAT_UNISTD_H
#define CODEX_COMPAT_UNISTD_H

#ifdef _WIN32
/* Windows has no POSIX unistd.h. The benchmark kernels that use this shim do
   not require POSIX declarations while extracting static representations. */
#else
/* Do not shadow the real POSIX declarations when this compatibility directory
   is present in a Linux compiler search path. */
#include_next <unistd.h>
#endif

#endif
