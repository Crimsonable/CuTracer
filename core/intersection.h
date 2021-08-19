#pragma once
#include "base.h"
#include "camera.h"
#include "scene.h"

namespace Tracer {

template <typename T>
__global__ void save(T *src, Camera camera, size_t width, size_t height) {
  size_t x = threadIdx.x;
  size_t y = blockIdx.x;
  size_t offset = y * 256 + x;
  x = offset % width;
  y = offset / width;
  offset = x + y * width;
  if (offset < width * height) {
    auto ray = camera.get_ray(Float(x) / width, Float(y) / height);
    auto t = Tvec3f(0, 0, 1);
    auto dd = Expblas::dot(ray.d, t);
    if (dd > cos(15.0f / 180 * Pi))
      src[offset] = TColor{1.0f, 1.0f, 1.0f, 1.0f};
    else
      src[offset] = TColor{0.0f, 0.0f, 0.0f, 1.0f};
  }
}

// template <Expblas::Device device>
// void trace(Expblas::Tensor<TColor, 2, device> &canvas, Camera &camera,
//           size_t width, size_t height) {
//  dim3 gridshape((canvas.shape.size() + 255) / 256, 1, 1);
//  dim3 blockshape(256, 1, 1);
//  save<<<gridshape, blockshape>>>(canvas.dataptr(), camera, width, height);
//}

template <Expblas::Device device> struct circle_draw_op {
  template <typename T>
  BlasForceInline BlasCudaFunc static void save(T &val, T _val, size_t y,
                                                size_t x, Camera &camera,
                                                size_t width, size_t height) {
    auto ray = camera.get_ray(Float(x) / width, Float(y) / height);
    auto dd = Expblas::dot(ray.d, Tvec3f(0, 0, 1));
    val = (dd > cos(15.0f / 180 * Pi)) ? TColor{1.0f, 0.5f, 0.5f, 1.0f}
                                       : TColor{0.0f, 0.5f, 0.0f, 1.0f};
  }
};

template <Expblas::Device device> struct TriangleIntersection_op {
  template <typename T>
  BlasForceInline BlasCudaConstruc static void
  save(T &val, T _val, size_t y, size_t x, Camera &camera, SceneGPUProxy &scene,
       size_t width, size_t height) {
    auto ray = camera.get_ray(Float(x) / width, Float(y) / height);
    bool hit = false;
    IntersectInfo info;
    for (int i = 0; i < scene.n; ++i) {
      for (int j = 0; j < scene.models[i].n; ++j) {
        BVHIntersect(ray, scene.models[i].tree[j].Internal,
                     scene.models[i].tree[j].vertex,
                     scene.models[i].tree[j].indice, info);
      }
    }
    val = hit ? TColor{1.0f, 0.5f, 0.5f, 1.0f} : TColor{0.0f, 0.5f, 0.0f, 1.0f};
  }
};

template <Expblas::Device device>
void trace(Expblas::Tensor<TColor, 2, device> &canvas, Camera &camera,
           size_t width, size_t height) {
  Expblas::ExpEngine<circle_draw_op<device>, TColor>::eval(
      &canvas, Expblas::MakeUnaryExp<OP::Unary::none>(canvas), camera, width,
      height);
}

template <Expblas::Device device>
void trace(Expblas::Tensor<TColor, 2, device> &canvas, Camera &camera,
           SceneGPUProxy &scene, size_t width, size_t height) {
  Expblas::ExpEngine<TriangleIntersection_op<device>, TColor>::eval(
      &canvas, Expblas::MakeUnaryExp<OP::Unary::none>(canvas), camera, scene,
      width, height);
}
} // namespace Tracer