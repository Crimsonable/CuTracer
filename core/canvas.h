#include "base.h"
#include "camera.h"
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
  Expblas::device_vector<SceneGPUProxy> gpu_scene;
  Expblas::device_vector<Camera> gpu_camera;
  Expblas::Shape<2> shape;
  GLuint PBO, texture;
  cudaGraphicsResource_t cudaResources;
  bool first = true;
};
} // namespace Tracer