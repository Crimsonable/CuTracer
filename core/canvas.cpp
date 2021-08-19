#include "canvas.h"
#include "gl_cuda_interop.h"

using namespace Tracer;

Tracer::Canvas::Canvas(Expblas::Shape<2> _shape, Camera *_camera, Scene *_scene)
    : shape(_shape), canvas(_shape), camera(_camera), scene(_scene) {
  canvas.data = new Expblas::DenseStorage<TColor, Expblas::Device::GPU>;
  canvas.data->stride = shape[0];
  canvas.data->size = shape.size();
}

void Tracer::Canvas::init() {
  CreateBuffer(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, PBO,
               sizeof(TColor) * shape[0] * shape[1]);
  Interop_BindBuffer(PBO, cudaResources);
  Interop_CreateTexture(texture, shape[0], shape[1]);
  scene_proxy = SceneGPUProxy(scene->n, scene->models.data());
}

void Tracer::Canvas::cudaDraw() {
  size_t _size = canvas.data->size * sizeof(TColor);
  Interop_LockCudaBufferPointer(canvas.data->dataptr, &_size, cudaResources);
  trace(canvas, *camera, scene_proxy, shape[0], shape[1]);
  Interop_UnlockCudaBufferPointer(cudaResources);
}

void Tracer::Canvas::submit() {
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, shape[0], shape[1], GL_RGBA, GL_FLOAT,
                  NULL);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void Tracer::Canvas::destory() { Expblas::freeSpace(canvas); }

void Tracer::Canvas::clear(TColor color) { Expblas::init(canvas, color); }
