#ifndef CODEX_COMPAT_UNISTD_H
#define CODEX_COMPAT_UNISTD_H

/* Minimal compatibility shim for Clang AST/LLVM IR generation on Windows.
   PolyBenchC includes <unistd.h>, but the benchmark kernels used here do not
   require POSIX declarations for static representation extraction. */

#endif
