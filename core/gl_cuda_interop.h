#pragma once
#include "base.h"

namespace Tracer {
void CreateBuffer(GLenum target, GLenum DrawHint, Tuint &gl_buffer, size_t size,
                  void *data = NULL);

void Interop_BindBuffer(Tuint gl_buffer, cudaGraphicsResource_t &cuda_buffer);

void Interop_CreateTexture(Tuint &gl_texture, size_t width, size_t height);

void Interop_UnlockCudaBufferPointer(cudaGraphicsResource_t &cuda_buffer);

template <typename T>
void Interop_LockCudaBufferPointer(T *&cuda_ptr, size_t *size,
                                   cudaGraphicsResource_t &cuda_buffer) {
  EXPBLAS_CUDA_CALL(cudaGraphicsMapResources(1, &cuda_buffer, 0));
  EXPBLAS_CUDA_CALL(cudaGraphicsResourceGetMappedPointer((void **)&cuda_ptr,
                                                         size, cuda_buffer));
}
} // namespace Tracer