#pragma once
#include "../base.h"

namespace Tracer {
inline BlasCudaConstruc Tuint ExpandBits(Tuint v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

inline BlasCudaConstruc Tuint CalMortonCode(const Tvec3f &v) {
  auto x = fminf(fmaxf((v[0]+0.5) * 1024.0f, 0.0f), 1023.0f);
  auto y = fminf(fmaxf((v[1]+0.5) * 1024.0f, 0.0f), 1023.0f);
  auto z = fminf(fmaxf((v[2]+0.5) * 1024.0f, 0.0f), 1023.0f);
  Tuint xx = ExpandBits((Tuint)x);
  Tuint yy = ExpandBits((Tuint)y);
  Tuint zz = ExpandBits((Tuint)z);
  return (xx * 4 + yy * 2 + zz);
}
} // namespace Tracer