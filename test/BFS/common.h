
#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef PARENT_BLOCK_SIZE
#define PARENT_BLOCK_SIZE 1024
#endif
#ifndef CHILD_BLOCK_SIZE
#define CHILD_BLOCK_SIZE 32
#endif

#define WARP_SIZE 32

void launch_kernel(unsigned int *d_costArray, unsigned int *d_edgeArray, unsigned int *d_edgeArrayAux, unsigned int numVerts, int iters, int* d_flag);

// workaround for OS X Snow Leopard w/ gcc 4.2.1 and CUDA 2.3a
// (undefined __sync_fetch_and_add)
#if defined(__APPLE__)
# if _GLIBCXX_ATOMIC_BUILTINS == 1
#undef _GLIBCXX_ATOMIC_BUILTINS
#endif // _GLIBC_ATOMIC_BUILTINS
#endif // __APPLE__

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// On Windows, if we call exit, our console may disappear,
// taking the error message with it, so prompt before exiting.
#if defined(_WIN32)
#define safe_exit(val)                          \
{                                               \
    cout << "Press return to exit\n";           \
    cin.get();                                  \
    exit(val);                                  \
}
#else
#define safe_exit(val) exit(val)
#endif

#define CHECK_CUDA_ERROR()                                                    \
{                                                                             \
    cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess)                                                   \
    {                                                                         \
        printf("error=%d name=%s at "                                         \
               "ln: %d\n  ",err,cudaGetErrorString(err),__LINE__);            \
        safe_exit(-1);                                                        \
    }                                                                         \
}

// Alternative macro to catch CUDA errors
#define CUDA_SAFE_CALL( call) do {                                            \
   cudaError err = call;                                                      \
   if (cudaSuccess != err) {                                                  \
       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",          \
           __FILE__, __LINE__, cudaGetErrorString( err) );                    \
       safe_exit(EXIT_FAILURE);                                               \
   }                                                                          \
} while (0)

// Alleviate aliasing issues
#define RESTRICT __restrict__

#endif

