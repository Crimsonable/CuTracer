#include "triangle.h"

using namespace Tracer;

BlasCudaConstruc bool
Tracer::Triangle::Intersect(const Ray &ray, const Tvec3f &v0, const Tvec3f &v1,
                            const Tvec3f &v2, IntersectInfo &info) {
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

BlasCudaConstruc TriangleVertexRef
Tracer::Triangle::GetVertex(VertexData *data, TriangleIndex *indice, Tuint id) {
  return TriangleVertexRef(data[indice[id].v1].position,
                           data[indice[id].v2].position,
                           data[indice[id].v3].position);
}
