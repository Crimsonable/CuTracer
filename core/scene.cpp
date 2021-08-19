#include "scene.h"

using namespace Tracer;

void Tracer::Scene::AddModel(const std::string &path) {
  models.push_back(new Model(path));
  n += 1;
}

Tracer::SceneGPUProxy::SceneGPUProxy(int _n, Model **_models) {
  n = _n;
  for (int i = 0; i < n; ++i)
    models[i] =
        ModelGPUProxy(_models[i]->n, _models[i]->tree_container.data(),
                      _models[i]->vertex.data(), _models[i]->indice.data());
}
