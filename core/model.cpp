#include "model.h"
#include "gl_cuda_interop.h"

using namespace Tracer;

void Tracer::Model::LoadModel(const std::string &path) {
  Assimp::Importer import;
  const aiScene *scene = import.ReadFile(
      path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals |
                aiProcess_FixInfacingNormals);

  if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
      !scene->mRootNode) {
    std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
    return;
  }
  directory = path.substr(0, path.find_last_of('/'));
  processNode(scene->mRootNode, scene);
}

VertexData *Tracer::Model::getMeshi(size_t idx) { return vertex[idx].first; }

Tuint *Tracer::Model::getIndicei(size_t idx) { return indice[idx].first; }

void Tracer::Model::processNode(aiNode *node, const aiScene *scene) {
  vertex.reserve(node->mNumMeshes);
  indice.reserve(node->mNumMeshes);
  n = node->mNumMeshes;
  std::vector<VertexData> tempbuffer_v;
  std::vector<Tuint> tempbuffer_i;
  for (Tuint i = 0; i < node->mNumMeshes; i++) {
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    vertex.push_back({nullptr, 0});
    indice.push_back({nullptr, 0});
    tempbuffer_v.reserve(mesh->mNumVertices);
    tempbuffer_v.reserve(mesh->mNumFaces);
    processMesh(mesh, scene, tempbuffer_v, tempbuffer_i);
    EXPBLAS_CUDA_CALL(cudaMalloc(&vertex.back().first,
                                 sizeof(VertexData) * tempbuffer_v.size()))
    EXPBLAS_CUDA_CALL(cudaMemcpy(vertex.back().first, tempbuffer_v.data(),
                                 sizeof(VertexData) * tempbuffer_v.size(),
                                 cudaMemcpyHostToDevice))
    EXPBLAS_CUDA_CALL(
        cudaMalloc(&indice.back().first, sizeof(Tuint) * tempbuffer_i.size()))
    EXPBLAS_CUDA_CALL(cudaMemcpy(indice.back().first, tempbuffer_i.data(),
                                 sizeof(Tuint) * tempbuffer_i.size(),
                                 cudaMemcpyHostToDevice))

    vertex.back().second = tempbuffer_v.size();
    indice.back().second = tempbuffer_i.size();
  }

  /*for (unsigned int i = 0; i < node->mNumChildren; i++) {
    processNode(node->mChildren[i], scene);
  }*/
}

void Tracer::Model::processMesh(aiMesh *mesh, const aiScene *scene,
                                std::vector<VertexData> &vertices,
                                std::vector<Tuint> &indices) {
  vertices.reserve(mesh->mNumVertices);
  indices.reserve(mesh->mNumFaces);

  for (int i = 0; i < mesh->mNumVertices; ++i) {
    VertexData vertex;
    Tvec3f vec;
    vec = Tvec3f(mesh->mVertices[i].x, mesh->mVertices[i].y,
                 mesh->mVertices[i].z);
    vertex.position = vec;
    vec = Tvec3f(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
    vertex.normal = vec;
    vertices.push_back(vertex);
  }
  for (int i = 0; i < mesh->mNumFaces; ++i) {
    aiFace face = mesh->mFaces[i];
    for (int i = 0; i < face.mNumIndices; ++i)
      indices.push_back(face.mIndices[i]);
  }
}

void Tracer::Model::BuildBVH() {
  tree_container.reserve(vertex.size());
  EXPBLAS_CUDA_KERNELPASTCHECK();

  for (int i = 0; i < vertex.size(); ++i) {
    tree_container.push_back(BVH::BVH<Expblas::Device::GPU>(
        size_t(indice[i].second / 3), vertex[i].first, indice[i].first));
  }
}

Tracer::Model::~Model() {
  for (int i = 0; i < vertex.size(); ++i) {
    cudaFree(vertex[i].first);
    cudaFree(indice[i].first);
  }
}

Tracer::Model::Model(const std::string &path) {
  LoadModel(path);
  BuildBVH();
}

Tracer::Model::Model(std::vector<VertexData> &vertices,
                     std::vector<Tuint> &indices, size_t nmesh, size_t nfaces) {
  vertex.reserve(nmesh);
  indice.reserve(nmesh);
  n = nmesh;
  for (int i = 0; i < nmesh; ++i) {
    vertex.push_back({nullptr, 0});
    indice.push_back({nullptr, 0});
  }
  BuildBVH();
}

Tracer::ModelGPUProxy::ModelGPUProxy(int _n,
                                     BVH::BVH<Expblas::Device::GPU> *bvh,
                                     VertexInfo *_vertex, IndiceInfo *indice) {
  n = _n;
  for (int i = 0; i < n; ++i)
    tree[i] = BVH::BVHPtr<Expblas::Device::GPU>(&bvh[i], _vertex[i].first,
                                                indice[i].first);
}
