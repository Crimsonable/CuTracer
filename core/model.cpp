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

VertexData *Tracer::Model::getMeshi(size_t idx) {
  VertexData *data;
  size_t size = sizeof(VertexData) * std::get<2>(VBO[idx]);
  Interop_LockCudaBufferPointer(data, &size, std::get<1>(VBO[idx]));
  return data;
}

Tuint *Tracer::Model::getIndicei(size_t idx) {
  Tuint *data;
  size_t size = sizeof(Tuint) * std::get<2>(EBO[idx]);
  Interop_LockCudaBufferPointer(data, &size, std::get<1>(EBO[idx]));
  return data;
}

void Tracer::Model::processNode(aiNode *node, const aiScene *scene) {
  VBO.reserve(node->mNumMeshes);
  EBO.reserve(node->mNumMeshes);
  n = node->mNumMeshes;
  vertex.reserve(node->mNumMeshes);
  indice.reserve(node->mNumMeshes);
  for (Tuint i = 0; i < node->mNumMeshes; i++) {
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    VBO.push_back({0, 0, 0});
    EBO.push_back({0, 0, 0});
    vertex.push_back(nullptr);
    indice.push_back(nullptr);
    std::vector<VertexData> vertices;
    std::vector<Tuint> indices;
    processMesh(mesh, scene, vertices, indices);
    BufferBind(vertices, indices, i, mesh[i].mNumVertices, mesh[i].mNumFaces);
    LoadCudaPtr(i);
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

void Tracer::Model::BufferBind(std::vector<VertexData> &vertices,
                               std::vector<Tuint> &indices, size_t i,
                               size_t nvertices, size_t nfaces) {
  CreateBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW, std::get<0>(VBO[i]),
               vertices.size() * sizeof(VertexData), vertices.data());
  Interop_BindBuffer(std::get<0>(VBO[i]), std::get<1>(VBO[i]));
  CreateBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW, std::get<0>(EBO[i]),
               indices.size() * sizeof(Tuint), indices.data());
  Interop_BindBuffer(std::get<0>(EBO[i]), std::get<1>(EBO[i]));

  std::get<2>(VBO[i]) = nvertices;
  std::get<2>(EBO[i]) = nfaces;
  node_size += nvertices;
  face_size += nfaces;
}

void Tracer::Model::LoadCudaPtr(size_t idx) {
  /*size_t v, i;
  v = std::get<2>(VBO[idx]) * sizeof(VertexData);
  i = std::get<2>(EBO[idx]) * sizeof(Tuint) * 3;*/
  size_t size;
  Interop_LockCudaBufferPointer(vertex[idx], &size, std::get<1>(VBO[idx]));
  Interop_LockCudaBufferPointer(indice[idx], &size, std::get<1>(EBO[idx]));
}

void Tracer::Model::BuildBVH() {
  tree_container.reserve(VBO.size());
  for (int i = 0; i < VBO.size(); ++i) {
    tree_container.emplace_back(std::get<2>(EBO[i]), vertex[i],
                                reinterpret_cast<TriangleIndex *>(indice[i]));
  }
}

Tracer::Model::~Model() {
  for (int i = 0; i < VBO.size(); ++i)
    glDeleteBuffers(1, &std::get<0>(VBO[i]));
  for (int i = 0; i < EBO.size(); ++i)
    glDeleteBuffers(1, &std::get<0>(EBO[i]));
}

Tracer::Model::Model(const std::string &path) {
  LoadModel(path);
  BuildBVH();
}

Tracer::Model::Model(std::vector<VertexData> &vertices,
                     std::vector<Tuint> &indices, size_t nmesh, size_t nfaces) {
  VBO.reserve(nmesh);
  EBO.reserve(nmesh);
  n = nmesh;
  for (int i = 0; i < nmesh; ++i) {
    VBO.push_back({0, 0, 0});
    EBO.push_back({0, 0, 0});
    BufferBind(vertices, indices, i, vertices.size(), nfaces);
    LoadCudaPtr(i);
  }
  BuildBVH();
}

Tracer::ModelGPUProxy::ModelGPUProxy(int _n,
                                     BVH::BVH<Expblas::Device::GPU> *bvh,
                                     VertexData **_vertex,
                                     TriangleIndex **indice) {
  n = _n;
  for (int i = 0; i < n; ++i)
    tree[i] = BVH::BVHPtr<Expblas::Device::GPU>(&bvh[i], _vertex[i], indice[i]);
}
