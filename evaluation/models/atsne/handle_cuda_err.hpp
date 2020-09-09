#pragma once
#include <cassert>
#include <cstdio>
// #include <cublas.h>

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define HANDLE_NULL(a)                                                           \
    {                                                                            \
        if (a == NULL) {                                                         \
            printf("Host memory failed in %s at line %d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }
#define CudaCheckError() HandleError(cudaGetLastError(), __FILE__, __LINE__)

// CURAND
#define CURAND_CALL(x)                                      \
    do {                                                    \
        if ((x) != CURAND_STATUS_SUCCESS) {                 \
            printf("Error at %s:%d\n", __FILE__, __LINE__); \
  /*return EXIT_FAILURE;*/}                                 \
    } while (0)

// CUBLAS
#ifndef cublasSafeCall
#define cublasSafeCall(err) __cublasSafeCall(err, __FILE__, __LINE__)
#endif

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line) {
    if (CUBLAS_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n", __FILE__, __LINE__, err);
        /*getch();*/ cudaDeviceReset();
        assert(0);
    }
}
