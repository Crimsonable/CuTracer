#include "base.h"

namespace Tracer {
struct TriangleIndex {
  Tuint v1, v2, v3;
};

struct TirangleData {
  VertexData v1, v2, v3;
};

struct TriangleVertexRef {
  BlasCudaConstruc TriangleVertexRef(const Tvec3f &_v1, const Tvec3f &_v2,
                                     const Tvec3f &_v3) {
    v1 = _v1;
    v2 = _v2;
    v3 = _v3;
  }

  BlasForceInline BlasCudaConstruc Tvec3f centroid() const {
    return 0.5f * (v1 + v2 + v3);
  }

  BlasCudaConstruc Bound3 bound() const { return Bound3().Union(v1, v2, v3); }

  Tvec3f v1, v2, v3;
};

struct Triangle {
  BlasCudaConstruc static bool Intersect(const Ray &ray, const Tvec3f &v0,
                                         const Tvec3f &v1, const Tvec3f &v2,
                                         IntersectInfo &info) {
    auto E1 = v1 - v0;
    auto E2 = v2 - v0;
    auto P = Expblas::cross(ray.d, E2);
    float det = Expblas::dot(E1, P);
    auto T = (2 * (Float(det > 0) - 0.5f)) * (ray.o - v0);
    /*if (det > 0)
      return;
    auto T = ray.o - v0;*/
    det = abs(det);
    info.uv[0] = Expblas::dot(T, P);
    if (info.uv[0] < 0)
      return false;
    auto Q = Expblas::cross(T, E1);
    info.uv[1] = Expblas::dot(ray.d, Q);
    if (info.uv[1] < 0 || info.uv[0] + info.uv[1] > det)
      return false;
    info.t = Expblas::dot(E2, Q);
    float invdet = 1.0f / (det + Delta);
    info.uv[0] *= invdet;
    info.uv[1] *= invdet;
    info.t *= invdet;
  }

  BlasCudaConstruc static TriangleVertexRef GetVertex(VertexData *data,
                                                      Tuint *indice, Tuint id) {
    return TriangleVertexRef(data[indice[id]].position,
                             data[indice[id + 1]].position,
                             data[indice[id + 2]].position);
  }
};
} // namespace Tracer