#pragma once
#include "../triangle.h"
#include "morton.cuh"

namespace Tracer {
namespace BVH {

struct Node {
  BlasCudaConstruc Node(const Bound3 &b, bool flag) : box(b), isLeafe(flag) {}
  BlasCudaConstruc Node(bool flag) : isLeafe(flag) {}

  virtual BlasCudaConstruc Node *getChildLeft() = 0;
  virtual BlasCudaConstruc Node *getChildRight() = 0;

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
  BlasCudaConstruc LeafeNode(const Bound3 &b, Tuint id, Tuint morton_)
      : Node(b, true), objId(id), morton(morton_) {}

  BlasCudaConstruc LeafeNode() : Node(true) {}

  BlasCudaConstruc Node *getChildLeft() { return nullptr; }

  BlasCudaConstruc Node *getChildRight() { return nullptr; }

  BlasCudaConstruc bool intersect(const Ray &ray, IntersectInfo &info,
                                  VertexData *vertex,
                                  TriangleIndex *indice) const {
    auto tri = Triangle::GetVertex(vertex, indice, objId);
    return Triangle::Intersect(ray, tri.v1, tri.v2, tri.v3, info);
  }

  BlasCudaConstruc constexpr bool operator<(const LeafeNode &other) const {
    return this->morton < other.morton;
  }

  Tuint objId, morton;
};

__global__ void GenMortonCode(VertexData *vertex, TriangleIndex *indice,
                              LeafeNode *l_nodes, Tuint n) {
  Tuint idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    auto tri = Triangle::GetVertex(vertex, indice, idx);
    Tuint morton = CalMortonCode(tri.Centroid());
    l_nodes[idx] = LeafeNode(tri.bound(), idx, morton);
  }
}

__device__ Tuint FindRange(Tuint start, int dir, LeafeNode *l_nodes) {
  Tuint min_clz = __clz(l_nodes[start].morton ^ l_nodes[start - dir].morton);
  Tuint l_max = 2;
  while (__clz(l_nodes[start].morton ^ l_nodes[start + l_max * dir].morton) >
         min_clz)
    l_max *= 2;

  Tuint l = 0;
  for (Tuint t = l_max / 2; t >= 1; t /= 2)
    if (__clz(l_nodes[start].morton ^ l_nodes[start + (l + t) * dir].morton) >
        min_clz)
      l += t;
  return l * dir + start;
}

__device__ Tuint FindSplit(Tuint start, Tuint end, int dir,
                           LeafeNode *l_nodes) {
  if (l_nodes[start].morton == l_nodes[end].morton)
    return (start + end) >> 1;
  Tuint s = 0;
  Tuint range = abs(int(end - start));
  Tuint range_clz = __clz(l_nodes[start].morton ^ l_nodes[end].morton);
  for (int t = range / 2; t >= 1; t /= 2)
    if (__clz(l_nodes[start].morton ^ l_nodes[start + (t + s) * dir].morton) >
        range_clz)
      s += t;
  return s * dir + start;
}

__global__ void build(VertexData *tri_vertex, TriangleIndex *tri_indice,
                      LeafeNode *l_nodes, InternalNode *i_nodes, Tuint n) {
  Tuint idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n - 1) {
    Tvec2ui range;
    int dir = 0;
    range[0] = idx;
    if (idx) {
      dir = __clz(l_nodes[idx].morton ^ l_nodes[idx + 1].morton) -
                        __clz(l_nodes[idx].morton ^ l_nodes[idx - 1].morton) >
                    0
                ? 1
                : -1;
      range[1] = FindRange(idx, dir, l_nodes);
    } else {
      range[1] = n - 2;
      dir = 1;
    }
    auto split = FindSplit(idx, range[1], dir, l_nodes);

    if (dir < 0) {
      auto temp = range[0];
      range[0] = range[1];
      range[1] = temp;
    }

    i_nodes[idx].range = range;
    i_nodes[idx].left =
        split == range[0] ? (Node *)&l_nodes[split] : (Node *)&i_nodes[split];
    i_nodes[idx].right = split + 1 == range[1] ? (Node *)&l_nodes[split + 1]
                                               : (Node *)&i_nodes[split + 1];
    i_nodes[idx].left->parent = &i_nodes[idx];
    i_nodes[idx].right->parent = &i_nodes[idx];
    i_nodes[idx].id = idx;
  }
}

// bottom-up method,for each leafe node in bvh,one thread find its parent node
// and make the atomic counter = 1,when another thread pass the parent
// node,execute the merge operation
__global__ void GenBBox(LeafeNode *l_nodes, InternalNode *i_nodes, Tuint *atom,
                        Tuint n) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    InternalNode *ptr = static_cast<InternalNode *>(l_nodes[idx].getParent());
    while (ptr && idx < n - 1) {
      if (atomicCAS(&atom[ptr->getId()], 0, 1) == 1) {
        ptr->box = ptr->box.Union(ptr->left->getBBox());
        ptr->box = ptr->box.Union(ptr->right->getBBox());

        ptr = static_cast<InternalNode *>(ptr->parent ? ptr->parent : nullptr);
      }
    }
  }
}

__device__ bool BVHIntersect(Ray &ray, InternalNode *bvh, VertexData *vertex,
                             TriangleIndex *indice, IntersectInfo &hit_info) {
  Node *stack[64];
  Node **ptr = stack;
  *ptr++ = nullptr;
  Node *node = bvh;
  Float t = 10000.0f;
  bool ishit = false;

  while (node) {
    Float temp_t;
    // call intersectTest to test if ray hit the bbox.
    bool temp_ishit = node->intersectTest(ray, temp_t);
    ptr--;
    if (temp_ishit && t > temp_t) {
      t = temp_t;
      // if node is LeafeNode, call intersect to get detail info.
      if (node->isLeafe) {
        IntersectInfo temp_info;
        temp_ishit = static_cast<LeafeNode *>(node)->intersect(ray, temp_info,
                                                               vertex, indice);
        // the hit which has smaller t become the closet hit point.
        hit_info =
            (temp_ishit) && (temp_info.t < hit_info.t) ? temp_info : hit_info;
        ishit = (temp_ishit) && (temp_info.t < hit_info.t) ? true : false;
      } else {
        // node is InternalNode.
        Node *left = node->getChildLeft();
        Node *right = node->getChildRight();
        // in stack.
        *++ptr = left;
        *++ptr = right;
      }
    }
    node = *ptr;
  }
  return ishit;
}

template <Expblas::Device device> class BVH;

template <> class BVH<Expblas::Device::GPU> {

public:
  BVH(size_t _n, VertexData *tri_vertex, TriangleIndex *tri_indice)
      : n(_n), leafe_nodes(_n), internal_nodes(_n - 1) {

    dim3 grid_shape((n + Expblas::CudaPars::SugThreadsPerBlock - 1) /
                        Expblas::CudaPars::SugThreadsPerBlock,
                    1, 1);
    dim3 block_shape(Expblas::CudaPars::SugThreadsPerBlock, 1, 1);
    GenMortonCode<<<grid_shape, block_shape>>>(
        tri_vertex, tri_indice, thrust::raw_pointer_cast(leafe_nodes.data()),
        n);

    thrust::sort(leafe_nodes.begin(), leafe_nodes.end(),
                 thrust::less<LeafeNode>());

    internal_nodes[0] = InternalNode(Bound3(), Tvec2ui{0, n - 1});

    grid_shape = dim3((n + Expblas::CudaPars::SugThreadsPerBlock - 2) /
                          Expblas::CudaPars::SugThreadsPerBlock,
                      1, 1);

    build<<<grid_shape, block_shape>>>(
        tri_vertex, tri_indice, thrust::raw_pointer_cast(leafe_nodes.data()),
        thrust::raw_pointer_cast(internal_nodes.data()), n);

    auto atom = thrust::device_vector<Tuint>(n, 0);

    GenBBox<<<grid_shape, block_shape>>>(
        thrust::raw_pointer_cast(leafe_nodes.data()),
        thrust::raw_pointer_cast(internal_nodes.data()),
        thrust::raw_pointer_cast(atom.data()), n);
  }

  size_t n;
  thrust::device_vector<LeafeNode> leafe_nodes;
  thrust::device_vector<InternalNode> internal_nodes;
};

template <Expblas::Device device> struct BVHPtr;

template <> struct BVHPtr<Expblas::Device::GPU> {
  BVHPtr(BVH<Expblas::Device::GPU> *bvh, VertexData *_vertex,
         TriangleIndex *_indice)
      : vertex(_vertex), indice(_indice) {
    leafe = thrust::raw_pointer_cast(bvh->leafe_nodes.data());
    Internal = thrust::raw_pointer_cast(bvh->internal_nodes.data());
    n = bvh->n;
  }
  BVHPtr() {}

  VertexData *vertex;
  TriangleIndex *indice;
  LeafeNode *leafe;
  InternalNode *Internal;
  size_t n;
};

} // namespace BVH
} // namespace Tracer