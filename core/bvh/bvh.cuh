#pragma once
#include "../triangle.h"
#include "cub/cub.cuh"
#include "morton.cuh"

namespace Tracer {
namespace BVH {
struct CubRadixSort {
  template <typename Tkey, typename Tval>
  static void SortPairs(Tkey *&key, Tval *&val, size_t n) {
    void *temp_buffer = NULL;
    size_t buffer_size;
    Tkey *kout = NULL;
    Tval *vout = NULL;
    cudaMalloc((void **)(&kout), n * sizeof(Tkey));
    cudaMalloc((void **)(&vout), n * sizeof(Tval));
    cub::DeviceRadixSort::SortPairs(temp_buffer, buffer_size, val, vout, key,
                                    kout, n);
    cudaMalloc((void **)(&temp_buffer), buffer_size);
    cub::DeviceRadixSort::SortPairs(temp_buffer, buffer_size, val, vout, key,
                                    kout, n);
    cudaFree(temp_buffer);
    key = kout;
    val = vout;
  }
};

struct Node {
  BlasCudaConstruc Node(const Bound3 &b, bool flag) : box(b), isLeafe(flag) {}
  BlasCudaConstruc Node(bool flag) : isLeafe(flag) {}

  // virtual BlasCudaConstruc Node *getChildLeft() = 0;
  // virtual BlasCudaConstruc Node *getChildRight() = 0;

  BlasCudaConstruc Node *getParent() { return parent; }
  BlasCudaConstruc Bound3 getBBox() const { return box; }
  BlasCudaConstruc Tuint getId() const { return id; }

  BlasCudaConstruc bool intersectTest(const Ray &ray, Float &t) const {
    return box.Intersect(ray, t);
  }

  Node *parent = nullptr;
  Bound3 box;
  Tuint id;
  bool isLeafe;
};

struct InternalNode : public Node {
  BlasCudaConstruc InternalNode(const Bound3 &b, const Tvec2ui &_range)
      : Node(b, false), range(_range) {}

  BlasCudaConstruc InternalNode() : Node(false) {}

  BlasCudaConstruc Node *getChildLeft() { return left; }

  BlasCudaConstruc Node *getChildRight() { return right; }

  Tvec2ui range;
  Node *left = nullptr, *right = nullptr;
};

struct LeafeNode : public Node {
  BlasCudaConstruc LeafeNode(const Bound3 &b, Tuint id)
      : Node(b, true), objId(id) {}

  BlasCudaConstruc LeafeNode() : Node(true) {}

  BlasCudaConstruc Node *getChildLeft() { return nullptr; }

  BlasCudaConstruc Node *getChildRight() { return nullptr; }

  BlasCudaConstruc bool intersect(const Ray &ray, IntersectInfo &info,
                                  VertexData *vertex, Tuint *indice) const {
    auto tri = Triangle::GetVertex(vertex, indice, objId);
    return Triangle::Intersect(ray, tri.v1, tri.v2, tri.v3, info);
  }

  /*BlasCudaConstruc constexpr bool operator<(const LeafeNode &other) const {
    return this->morton < other.morton;
  }*/

  Tuint objId;
  // Tuint morton;
};

__global__ void GenMortonCode(VertexData *vertex, Tuint *indice,
                              LeafeNode *l_nodes, Tuint *morton_code, Tuint n);

__device__ Tuint FindRange(Tuint start, int dir, Tuint *morton_code);

__device__ Tuint FindSplit(Tuint start, Tuint end, int dir, Tuint *morton_code);

__global__ void build(LeafeNode *l_nodes, InternalNode *i_nodes,
                      Tuint *morton_code, Tuint);

// bottom-up method,for each leafe node in bvh,one thread find its parent node
// and make the atomic counter = 1,when another thread pass the parent
// node,execute the merge operation
__global__ void GenBBox(LeafeNode *l_nodes, InternalNode *i_nodes, Tuint *atom,
                        Tuint n);

__device__ bool BVHIntersect(Ray &ray, InternalNode *bvh, VertexData *vertex,
                             Tuint *indice, IntersectInfo &hit_info);

__global__ void DebugEntry(LeafeNode *l_nodes, Tuint *m_code);

__global__ void DebugEntry(VertexData *vertex, Tuint *indice,
                           LeafeNode *l_nodes, Tuint *morton_code, Tuint n);

template <Expblas::Device device> class BVH;

template <> class BVH<Expblas::Device::GPU> {

public:
  BVH(size_t _n, VertexData *tri_vertex, Tuint *tri_indice)
      : n(_n), morton_code(_n), leafe_nodes(_n), internal_nodes(_n - 1) {

    /*dim3 grid_shape((n + Expblas::CudaPars::SugThreadsPerBlock - 1) /
                        Expblas::CudaPars::SugThreadsPerBlock,
                    1, 1);
    dim3 block_shape(Expblas::CudaPars::SugThreadsPerBlock, 1, 1);*/
    dim3 grid_shape(1, 1, 1);
    dim3 block_shape(1, 1, 1);
    DebugEntry<<<16, 1>>>(leafe_nodes.dataptr(), morton_code.dataptr());
    printf("end\n");
    DebugEntry<<<16, 1>>>(tri_vertex, tri_indice, leafe_nodes.dataptr(),
                          morton_code.dataptr(), n);
    printf("end\n");
    cudaDeviceSynchronize();
    EXPBLAS_CUDA_KERNELPASTCHECK();
    GenMortonCode<<<grid_shape, block_shape>>>(tri_vertex, tri_indice,
                                               leafe_nodes.dataptr(),
                                               morton_code.dataptr(), n);
    cudaDeviceSynchronize();
    EXPBLAS_CUDA_KERNELPASTCHECK();
    printf("%i", sizeof(TriangleIndex));
    auto l_cpu = Expblas::Tensor<Tuint, 2, Expblas::Device::CPU>(
        Expblas::Shape<2>{1, 3851});
    Expblas::allocSpace(l_cpu);
    Expblas::Copy(l_cpu, morton_code);
    EXPBLAS_CUDA_KERNELPASTCHECK();

    EXPBLAS_CUDA_KERNELPASTCHECK();
    auto temp_leafe = leafe_nodes;
    auto temp_morton = morton_code;
    CubRadixSort::SortPairs(temp_leafe.data->dataptr, temp_morton.data->dataptr,
                            n);
    Expblas::freeSpace(leafe_nodes);
    Expblas::freeSpace(morton_code);
    leafe_nodes = temp_leafe;
    morton_code = temp_morton;
    EXPBLAS_CUDA_KERNELPASTCHECK();

    internal_nodes.SetValue(0, InternalNode(Bound3(), Tvec2ui{0, n - 1}));

    grid_shape = dim3((n + Expblas::CudaPars::SugThreadsPerBlock - 2) /
                          Expblas::CudaPars::SugThreadsPerBlock,
                      1, 1);

    build<<<grid_shape, block_shape>>>(leafe_nodes.dataptr(),
                                       internal_nodes.dataptr(),
                                       morton_code.dataptr(), n);

    auto atom = Expblas::device_vector<Tuint>(n, 0);

    GenBBox<<<grid_shape, block_shape>>>(
        leafe_nodes.dataptr(), internal_nodes.dataptr(), atom.dataptr(), n);
  }

  size_t n;
  Expblas::device_vector<Tuint> morton_code;
  Expblas::device_vector<LeafeNode> leafe_nodes;
  Expblas::device_vector<InternalNode> internal_nodes;
};

template <Expblas::Device device> struct BVHPtr;

template <> struct BVHPtr<Expblas::Device::GPU> {
  BVHPtr(BVH<Expblas::Device::GPU> *bvh, VertexData *_vertex, Tuint *_indice)
      : vertex(_vertex), indice(_indice) {
    leafe = bvh->leafe_nodes.dataptr();
    Internal = bvh->internal_nodes.dataptr();
    n = bvh->n;
  }
  BVHPtr() {}

  VertexData *vertex;
  Tuint *indice;
  LeafeNode *leafe;
  InternalNode *Internal;
  size_t n;
};

} // namespace BVH
} // namespace Tracer