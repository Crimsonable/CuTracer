#pragma once
#define USE_CUDA
#include "../Expblas/ftensor.h"
#include "../Expblas/tensor.h"
#include "../GL/GLheaders.h"
#include <cmath>
#include <cuda/cuda_gl_interop.h>
//#include <cuda/thrust/device_vector.h>
//#include <cuda/thrust/sort.h>
//#include <thrust/device_ptr.h>
#include <vector>
//#include <cuda/thrust/host_vector.h>

#define Pi 3.14159265358979323846
#define Delta 1e-6
#define MAX_MODELS 5
#define MAX_MESHS 5

namespace Expblas {
template <typename T> class device_vector : public Tensor<T, 2, Device::GPU> {
public:
  device_vector(size_t n) : Tensor<T, 2, Device::GPU>(Shape<2>({1, n})) {
    allocSpace(*this);
  }

  device_vector(size_t n, T val) : Tensor<T, 1, Device::GPU>(Shape<2>({1, n})) {
    allocSpace(*this);
    init(*this, val);
  }

  device_vector() : Tensor<T, 2, Device::GPU>() {}

  //~device_vector() { freeSpace(*this); }

  void SetValue(size_t idx, T src) {
    cudaMemcpy(this->dataptr() + idx, &src, 1, cudaMemcpyHostToDevice);
  }

  /*T operator[](size_t idx) const { return this->eval(0, idx); }

  T &operator[](size_t idx) { return this->eval_ref(0, idx); }*/

  void reserve(size_t n) {
    this->shape = Shape<2>({1, n});
    allocSpace(*this);
  }
};
} // namespace Expblas

namespace Tracer {
using Float = float;
using Tuint = unsigned int;

using Tvec2i = Expblas::FTensor<int, 2>;
using Tvec2ui = Expblas::FTensor<Tuint, 2>;
using Tvec2f = Expblas::FTensor<Float, 2>;
using Tvec3f = Expblas::FTensor<Float, 3>;
using Tvec4f = Expblas::FTensor<Float, 4>;
using Tvec4i = Expblas::FTensor<int, 4>;
using Tmat4f = Expblas::FTensor<Float, 4, 4>;
using TColor = Tvec4f;

struct Ray {
  BlasCudaFunc Ray() {}

  BlasCudaFunc Ray(Tvec3f _o, Tvec3f _d) : o(_o), d(Expblas::normal(_d)) {}

  BlasCudaFunc Tvec3f at(Float t) const { return t * d + o; }

  Tvec3f o, d;
};

template <int N> struct Bound;

template <> struct Bound<3> {
  BlasCudaConstruc Bound(const Tvec3f &pmin, const Tvec3f &pmax)
      : pMin(pmin), pMax(pmax) {}

  BlasCudaConstruc Bound() : pMax{0, 0, 0}, pMin{0, 0, 0} {}

  BlasCudaConstruc Bound<3> Union(const Bound<3> &_b) const {
    Bound<3> res;
    res.pMin = Expblas::lowerBound(pMin, _b.pMin);
    res.pMax = Expblas::upperBound(pMax, _b.pMax);
    return res;
  }

  BlasCudaConstruc Bound<3> Union(const Tvec3f &v1, const Tvec3f &v2,
                                  const Tvec3f &v3) const {
    Bound<3> res;
    res.pMin = Expblas::lowerBound(Expblas::lowerBound(v1, v2), v3);
    res.pMax = Expblas::upperBound(Expblas::upperBound(v1, v2), v3);
    return res;
  }

  BlasCudaConstruc bool Intersect(const Ray &ray, Float &t) const {
    Tvec3f invd = 1.0f / ray.d;
    Tvec3f tMin = (pMin - ray.o) * ray.d;
    Tvec3f tMax = (pMax - ray.o) * ray.d;
    tMin = Expblas::lowerBound(tMin, tMax);
    tMax = Expblas::upperBound(tMin, tMax);
    t = std::min(tMin[0], std::min(tMin[1], tMin[2]));
    return t < std::max(tMin[0], std::max(tMin[1], tMin[2]));
  }

  Tvec3f pMin, pMax;
};

using Bound3 = Bound<3>;

struct VertexData {
  Tvec3f position;
  Tvec3f normal;
};

struct IntersectInfo {
  Float t = 100000.0f;
  Tuint id;
  Tvec3f normal;
  Tvec2f uv;
};

using VertexInfo = std::pair<VertexData *, size_t>;
using IndiceInfo = std::pair<Tuint *, size_t>;

} // namespace Tracer