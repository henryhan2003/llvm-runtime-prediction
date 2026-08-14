#ifdef __cplusplus
extern "C" {
#endif

// Local utility declarations. This file deliberately avoids the CUDA SDK's
// cuda.h name so Clang can find the real toolkit header in CUDA mode.

#include <stdio.h>

void setdevice(void);
void checkCUDAError(const char *msg);

#ifdef __cplusplus
}
#endif
