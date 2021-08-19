#include "base.h"

namespace Tracer {
struct TriangleIndex {
  Tuint v1, v2, v3;
};

struct TirangleData {
  VertexData v1, v2, v3;
};

struct TriangleVertexRef {
  TriangleVertexRef(const Tvec3f &_v1, const Tvec3f &_v2, const Tvec3f &_v3) {
    v1 = _v1;
    v2 = _v2;
    v3 = _v3;
  }

  BlasForceInline BlasCudaConstruc Tvec3f Centroid() const {
    return 0.5f * (v1 + v2 + v3);
  }

  BlasCudaConstruc Bound3 bound() const { return Bound3().Union(v1, v2, v3); }

  Tvec3f v1, v2, v3;
};

struct Triangle {
  BlasCudaConstruc static bool Intersect(const Ray &ray, const Tvec3f &v0,
                                         const Tvec3f &v1, const Tvec3f &v2,
                                         IntersectInfo &info);

  BlasCudaConstruc static TriangleVertexRef
  GetVertex(VertexData *data, TriangleIndex *indice, Tuint id);
};
} // namespace Tracer