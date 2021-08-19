#include "gl_cuda_interop.h"

using namespace Tracer;

void Tracer::CreateBuffer(GLenum target, GLenum DrawHint, Tuint &gl_buffer,
                          size_t size, void *data) {
  glGenBuffers(1, &gl_buffer);
  glBindBuffer(target, gl_buffer);
  glBufferData(target, size, data, DrawHint);
  glBindBuffer(target, 0);
}

void Tracer::Interop_BindBuffer(Tuint gl_buffer,
                                cudaGraphicsResource_t &cuda_buffer) {
  glBindBuffer(GL_ARRAY_BUFFER, gl_buffer);
  EXPBLAS_CUDA_CALL(cudaGraphicsGLRegisterBuffer(
      &cuda_buffer, gl_buffer, cudaGraphicsRegisterFlagsNone));
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Tracer::Interop_CreateTexture(Tuint &gl_texture, size_t width,
                                   size_t height) {
  glGenTextures(1, &gl_texture);
  glBindTexture(GL_TEXTURE_2D, gl_texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT,
               NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glBindTexture(GL_TEXTURE_2D, 0);
}

void Tracer::Interop_UnlockCudaBufferPointer(
    cudaGraphicsResource_t &cuda_buffer) {
  EXPBLAS_CUDA_CALL(cudaGraphicsUnmapResources(1, &cuda_buffer, 0));
}
