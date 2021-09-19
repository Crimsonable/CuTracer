#pragma once
#include "model.h"

namespace Tracer {
class Scene {
public:
  Scene() {}
  void AddModel(const std::string &path);

public:
  int n = 0;
  std::vector<Model *> models;
};

struct SceneGPUProxy {
  SceneGPUProxy(int n, Model **_models);
  SceneGPUProxy() {}

  int n;
  ModelGPUProxy models[MAX_MODELS];
};

} // namespace Tracer