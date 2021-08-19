#pragma once
//#include "mesh.h"
#include "./bvh/bvh.cuh"
#include "base.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

namespace Tracer {
class Model {
  using ResourcesPair = std::tuple<Tuint, cudaGraphicsResource_t, size_t>;

  void processNode(aiNode *node, const aiScene *scene);
  void processMesh(aiMesh *mesh, const aiScene *scene,
                   std::vector<VertexData> &vertices,
                   std::vector<Tuint> &indices);
  void BufferBind(std::vector<VertexData> &vertices,
                  std::vector<Tuint> &indices, size_t i, size_t nvertices,
                  size_t nfaces);
  void LoadCudaPtr(size_t idx);
  void BuildBVH();

public:
  Model(const std::string &path);
  Model(std::vector<VertexData> &vertices, std::vector<Tuint> &indices,
        size_t nmesh, size_t nfaces);
  ~Model();
  void LoadModel(const std::string &path);
  VertexData *getMeshi(size_t idx);
  Tuint *getIndicei(size_t idx);

  std::vector<ResourcesPair> VBO, EBO;
  std::vector<VertexData *> vertex;
  std::vector<TriangleIndex *> indice;
  std::vector<BVH::BVH<Expblas::Device::GPU>> tree_container;
  std::string directory;
  size_t node_size, face_size;
  int n;
};

struct ModelGPUProxy {
  ModelGPUProxy(int n, BVH::BVH<Expblas::Device::GPU> *bvh,
                VertexData **_vertex, TriangleIndex **indice);
  ModelGPUProxy() {}

  int n;
  BVH::BVHPtr<Expblas::Device::GPU> tree[MAX_MESHS];
};

} // namespace Tracer