#pragma once
#include <assert.h>
#include <cmath>
#include <cuda/cuda_runtime.h>
#include <cuda/device_launch_parameters.h>
#include <iostream>
#include <memory>
#include <new>
#include <omp.h>

#define EXPBLAS_CUDA_CALL(call)                                                \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("ERROR: %s:%d,", __FILE__, __LINE__);                             \
      printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));         \
    }                                                                          \
  }

#define EXPBLAS_CUDA_KERNELPASTCHECK()                                         \
  do {                                                                         \
    cudaError error = cudaPeekAtLastError();                                   \
    printf("error reason:%s\n", cudaGetErrorString(error));                    \
  } while (0)

#define EXP_ASSERT(condition, message)                                         \
  {                                                                            \
    if (!condition) {                                                          \
      printf("Error: %s", message);                                            \
      abort();                                                                 \
    }                                                                          \
  }

namespace Expblas {
enum class TensorState { Dynamic, Static };
enum class Device { GPU, CPU };
enum class OperatorType {
  container = 1,
  keepDim = 3,
  changeDim = 7,
  advance = 15
};
enum class Arch { AVX2 = 1, SSE = 3, Scalar = 7 };
enum class TransportDir { LocalToDevice, LocalToLocal };

#define USE_CUDA
#define BlasForceInline __forceinline

#ifdef USE_CUDA
//#define CUDA_ALIGN_ALLOC
#define BlasCudaFunc __device__ __host__ inline
#define BlasCudaConstruc __device__ __host__
#else
#define BlasCudaFunc
#endif // USE_CUDA

#define BlasUseSIMD false
#define BlasDefaultArch Arch::SSE

} // namespace Expblas