#pragma once
#include "base.h"

namespace Tracer {
class Camera {
public:
  Camera(Tvec3f lookfrom = Tvec3f(0, 0, 2), Tvec3f lookat = Tvec3f(0, 0, 0),
         Tvec3f vup = Tvec3f(0, 1, 0),
         Float vfov = 40, // vertical field-of-view in degrees
         Float aspect_ratio = 1, Float aperture = 0, Float focus_dist = 10,
         Float _time0 = 0, Float _time1 = 0) {
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
    lower_left_corner =
        origin - 0.5f * (horizontal + vertical) - focus_dist * w;

    lens_radius = aperture / 2;
    time0 = _time0;
    time1 = _time1;
  }

  BlasCudaConstruc Ray get_ray(Float s, Float t) const {
    Tvec3f rd = lens_radius;
    Tvec3f offset = rd[0] * u + rd[1] * v;
    return Ray(origin + offset, lower_left_corner + s * horizontal +
                                    t * vertical - origin - offset);
  }

private:
  Tvec3f origin;
  Tvec3f lower_left_corner;
  Tvec3f horizontal;
  Tvec3f vertical;
  Tvec3f u, v, w;
  double lens_radius;
  double time0, time1; // shutter open/close times
};
} // namespace Tracer