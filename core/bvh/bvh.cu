#pragma once
#include "bvh.cuh"
using namespace Tracer;

__global__ void Tracer::BVH::GenMortonCode(VertexData *vertex, Tuint *indice,
                                           LeafeNode *l_nodes,
                                           Tuint *morton_code, Tuint n) {
  Tuint idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    auto tri = Triangle::GetVertex(vertex, indice, idx);
    Tuint morton = CalMortonCode(tri.centroid());
    l_nodes[idx] = LeafeNode(tri.bound(), idx);
    morton_code[idx] = morton;
    printf("%i", morton_code[idx]);
  }
}

__device__ Tuint Tracer::BVH::FindRange(Tuint start, int dir,
                                        Tuint *morton_code) {
  Tuint min_clz = __clz(morton_code[start] ^ morton_code[start - dir]);
  Tuint l_max = 2;
  while (__clz(morton_code[start] ^ morton_code[start + l_max * dir]) > min_clz)
    l_max *= 2;

  Tuint l = 0;
  for (Tuint t = l_max / 2; t >= 1; t /= 2)
    if (__clz(morton_code[start] ^ morton_code[start + (l + t) * dir]) >
        min_clz)
      l += t;
  return l * dir + start;
}

__device__ Tuint Tracer::BVH::FindSplit(Tuint start, Tuint end, int dir,
                                        Tuint *morton_code) {
  if (morton_code[start] == morton_code[end])
    return (start + end) >> 1;
  Tuint s = 0;
  Tuint range = abs(int(end - start));
  Tuint range_clz = __clz(morton_code[start] ^ morton_code[end]);
  for (int t = range / 2; t >= 1; t /= 2)
    if (__clz(morton_code[start] ^ morton_code[start + (t + s) * dir]) >
        range_clz)
      s += t;
  return s * dir + start;
}

__global__ void Tracer::BVH::build(LeafeNode *l_nodes, InternalNode *i_nodes,
                                   Tuint *morton_code, Tuint n) {
  Tuint idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n - 1) {
    Tvec2ui range;
    int dir = 0;
    range[0] = idx;
    if (idx) {
      dir = __clz(morton_code[idx] ^ morton_code[idx + 1]) -
                        __clz(morton_code[idx] ^ morton_code[idx - 1]) >
                    0
                ? 1
                : -1;
      range[1] = FindRange(idx, dir, morton_code);
    } else {
      range[1] = n - 2;
      dir = 1;
    }
    auto split = FindSplit(idx, range[1], dir, morton_code);

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
__global__ void Tracer::BVH::GenBBox(LeafeNode *l_nodes, InternalNode *i_nodes,
                                     Tuint *atom, Tuint n) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    InternalNode *ptr = static_cast<InternalNode *>(l_nodes[idx].getParent());
    while (ptr && idx < n - 1) {
      if (atomicCAS(&atom[ptr->getId()], 0, 1) == 1) {
        ptr->box = ptr->box.Union(ptr->left->getBBox());
        ptr->box = ptr->box.Union(ptr->right->getBBox());
        ptr = static_cast<InternalNode *>(ptr->parent ? ptr->parent : nullptr);
      } else
        return;
    }
  }
}

__device__ bool Tracer::BVH::BVHIntersect(Ray &ray, InternalNode *bvh,
                                          VertexData *vertex, Tuint *indice,
                                          IntersectInfo &hit_info) {
  Node *stack[64];
  Node **ptr = stack;
  *ptr++ = nullptr;
  Node *node = bvh;
  Float t = 10000.0f;
  bool ishit = false;

  while (node) {
    Float temp_t;
    // call intersectTest to test if the ray hit the bbox.
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
        Node *left = static_cast<InternalNode *>(node)->getChildLeft();
        Node *right = static_cast<InternalNode *>(node)->getChildRight();
        // in stack.
        *++ptr = left;
        *++ptr = right;
      }
    }
    node = *ptr;
  }
  return ishit;
}

__global__ void Tracer::BVH::DebugEntry(LeafeNode *l_nodes, Tuint *m_code) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id < 3851)
    printf("%i", l_nodes[id].objId);
  return;
}

__global__ void Tracer::BVH::DebugEntry(VertexData *vertex, Tuint *indice,
                                        LeafeNode *l_nodes, Tuint *morton_code,
                                        Tuint n) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < 3851) {
    auto tri = Triangle::GetVertex(vertex, indice, idx);
    Tuint morton = CalMortonCode(tri.centroid());
    auto l = tri.bound();
    printf("%i\n", 1);
  }
}