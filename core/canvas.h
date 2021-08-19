#include "base.h"
#include "intersection.h"
#include "scene.h"

namespace Tracer {
struct Canvas {
  Canvas(Expblas::Shape<2> _shape, Camera *_camera, Scene *_scene);
  void init();
  void cudaDraw();
  void submit();
  void destory();
  void clear(TColor color);

  Expblas::Tensor<TColor, 2, Expblas::Device::GPU> canvas;
  Camera *camera;
  Scene *scene;
  SceneGPUProxy scene_proxy;
  Expblas::Shape<2> shape;
  GLuint PBO, texture;
  cudaGraphicsResource_t cudaResources;
  bool first = true;
};
} // namespace Tracer