#include "camera.h"

using namespace Tracer;

BlasCudaConstruc Tracer::Camera::Camera(Tvec3f lookfrom, Tvec3f lookat,
                                        Tvec3f vup, Float vfov,
                                        Float aspect_ratio, Float aperture,
                                        Float focus_dist, Float _time0,
                                        Float _time1) {
  Float theta = vfov / 180.0f * Pi;
  Float h = tan(theta / 2);
  Float viewport_height = 2.0 * h;
  Float viewport_width = aspect_ratio * viewport_height;

  w = Expblas::normal(lookfrom - lookat);
  u = Expblas::normal(cross(vup, w));
  v = cross(w, u);

  origin = lookfrom;
  horizontal = focus_dist * viewport_width * u;
  vertical = focus_dist * viewport_height * v;
  lower_left_corner = origin - 0.5f * (horizontal + vertical) - focus_dist * w;

  lens_radius = aperture / 2;
  time0 = _time0;
  time1 = _time1;
}

BlasCudaConstruc Ray Tracer::Camera::get_ray(Float s, Float t) const {
  Tvec3f rd = lens_radius;
  Tvec3f offset = rd[0] * u + rd[1] * v;
  return Ray(origin + offset, lower_left_corner + s * horizontal +
                                  t * vertical - origin - offset);
}