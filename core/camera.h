#pragma once
#include "base.h"

namespace Tracer {
class Camera {
public:
  BlasCudaConstruc Camera(Tvec3f lookfrom = Tvec3f(0, 0, 2),
                          Tvec3f lookat = Tvec3f(0, 0, 0),
                          Tvec3f vup = Tvec3f(0, 1, 0),
                          Float vfov = 40, // vertical field-of-view in degrees
                          Float aspect_ratio = 1, Float aperture = 0,
                          Float focus_dist = 10, Float _time0 = 0,
                          Float _time1 = 0);

  BlasCudaConstruc Ray get_ray(Float s, Float t) const;

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