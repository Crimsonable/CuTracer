#pragma once
#include "base.h"
#include "exp_engine.h"
#include "expbase.h"
#include "shape.h"
#include "storage.h"

namespace Expblas {
template <typename T, size_t... shapes>
class FTensor : public TensorBase<FTensor<T, shapes...>, T, Device::CPU> {
public:
  BlasCudaConstruc FTensor() {}

  template <typename... Args> BlasCudaConstruc FTensor(Args... args) {
    Meta::array_init(data, args...);
  }

  template <typename Exp, OperatorType exp_type>
  BlasForceInline auto &operator=(const ExpBase<Exp, T, exp_type> &exp) {
    return this->assign(exp.derived_to());
  }

  BlasForceInline T eval(size_t y, size_t x) const {
    return data[y * Meta::get_last(shapes...) + x];
  }

  BlasForceInline T eval(size_t idx) const { return data[idx]; }

  BlasForceInline T &eval_ref(size_t y, size_t x) {
    return data[y * Meta::get_last(shapes...) + x];
  }

  BlasForceInline T &eval_ref(size_t idx) { return data[idx]; }

  BlasForceInline T eval_broadcast(size_t y, size_t x) const {
    return data[y * Meta::get_last(shapes...) + x];
  }

  BlasForceInline BlasCudaFunc T operator[](size_t idx) const {
    return data[idx];
  }

  BlasForceInline BlasCudaFunc T &operator[](size_t idx) { return data[idx]; }

  BlasForceInline T *dataptr() const { return const_cast<T *>(data); }

  BlasForceInline T *dataptr() { return data; }

  BlasForceInline constexpr bool AllocCheck() const { return true; }

  BlasForceInline constexpr size_t stride() const {
    return Meta::get_last(shapes...);
  }

  BlasForceInline constexpr auto get_shape() const {
    return Shape<sizeof...(shapes)>{shapes...};
  }

  T data[Meta::FoldMul_naive(shapes...)]{T(0)};
};

namespace Fixed {
template <typename T, size_t _N>
BlasForceInline BlasCudaConstruc T dot_base_imp(const FTensor<T, _N> &v1,
                                                const FTensor<T, _N> &v2) {
  T res = 0;
  Meta::LoopUnroll<_N - 1>(
      [] __device__ __host__(size_t N, T & res, const FTensor<T, _N> &v1,
                             const FTensor<T, _N> &v2) {
        res += v1.data[N] * v2.data[N];
      },
      res, v1, v2);
  return res;
}

template <typename Op, typename T, size_t... shapes>
BlasForceInline BlasCudaConstruc FTensor<T, shapes...>
element_wise_op(const FTensor<T, shapes...> &v1,
                const FTensor<T, shapes...> &v2) {
  auto res = FTensor<T, shapes...>();
  Meta::LoopUnroll<Meta::FoldMul_naive(shapes...) - 1>(
      [] __device__ __host__(size_t N, FTensor<T, shapes...> & res,
                             const FTensor<T, shapes...> &v1,
                             const FTensor<T, shapes...> &v2) {
        res.data[N] = Op::eval(v1.data[N], v2.data[N]);
      },
      res, v1, v2);
  return res;
}

template <typename Op, typename T, size_t... shapes>
BlasForceInline BlasCudaConstruc FTensor<T, shapes...>
element_wise_op(const FTensor<T, shapes...> &v1, T v2) {
  auto res = FTensor<T, shapes...>();
  Meta::LoopUnroll<Meta::FoldMul_naive(shapes...) - 1>(
      [] __device__ __host__(size_t N, FTensor<T, shapes...> & res,
                             const FTensor<T, shapes...> &v1,
                             T &v2) { res.data[N] = Op::eval(v1.data[N], v2); },
      res, v1, v2);
  return res;
}

template <typename T, size_t _N>
BlasForceInline BlasCudaConstruc T
dot_base_naive_imp(const FTensor<T, _N> &v1, const FTensor<T, _N> &v2) {
  T res = 0;
  for (int i = 0; i < _N; ++i)
    res += v1[i] * v2[i];
  return res;
}

template <typename Op, typename T, size_t... shapes>
BlasForceInline BlasCudaConstruc FTensor<T, shapes...>
element_wise_naive_op(const FTensor<T, shapes...> &v1,
                      const FTensor<T, shapes...> &v2) {
  auto res = FTensor<T, shapes...>();
  for (int i = 0; i < Meta::FoldMul_naive(shapes...); ++i)
    res[i] = Op::eval(v1[i], v2[i]);
  return res;
}

template <typename Op, typename T, size_t... shapes>
BlasForceInline BlasCudaConstruc FTensor<T, shapes...>
element_wise_naive_op(const FTensor<T, shapes...> &v1, T v2) {
  auto res = FTensor<T, shapes...>();
  for (int i = 0; i < Meta::FoldMul_naive(shapes...); ++i)
    res[i] = Op::eval(v1[i], v2);
  return res;
}
} // namespace Fixed

template <size_t Ndst, size_t Nsrc, typename T,
          typename = std::enable_if_t<(Ndst <= Nsrc)>>
BlasForceInline BlasCudaConstruc FTensor<T, Ndst> *
FTensorDownCast(FTensor<T, Nsrc> &src) {
  return reinterpret_cast<FTensor<T, Ndst> *>(&src);
}

template <typename T, size_t N1, size_t N2,
          typename = std::enable_if_t<(N1 > 2) && (N2 > 2)>>
BlasForceInline BlasCudaConstruc auto cross(const FTensor<T, N1> &v1,
                                            const FTensor<T, N2> &v2) {
  auto res = FTensor<T, 3>();
  res[0] = v1[1] * v2[2] - v1[2] * v2[1];
  res[1] = v1[0] * v2[2] - v1[2] * v2[0];
  res[2] = v1[0] * v2[1] - v1[1] * v2[0];
  return res;
}

template <typename T, size_t N>
BlasForceInline BlasCudaConstruc auto dot(const FTensor<T, N> &v1,
                                          const FTensor<T, N> &v2) {
  return Fixed::dot_base_imp(v1, v2);
}

template <typename T, size_t... shapes>
BlasForceInline BlasCudaConstruc auto
operator+(const FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  return Fixed::element_wise_op<OP::Binary::plus>(v1, v2);
}

template <typename T, size_t... shapes>
BlasForceInline BlasCudaConstruc auto
operator-(const FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  return Fixed::element_wise_op<OP::Binary::minus>(v1, v2);
}

template <typename T, size_t... shapes>
BlasForceInline BlasCudaConstruc auto
operator*(const FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  return Fixed::element_wise_op<OP::Binary::mul>(v1, v2);
}

template <typename T, size_t... shapes>
BlasForceInline BlasCudaConstruc auto
operator/(const FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  return Fixed::element_wise_op<OP::Binary::div>(v1, v2);
}

template <typename T, size_t... shapes>
BlasForceInline BlasCudaConstruc auto
operator+(T v2, const FTensor<T, shapes...> &v1) {
  return Fixed::element_wise_op<OP::Binary::plus>(v1, v2);
}

template <typename T, size_t... shapes>
BlasForceInline BlasCudaConstruc auto
operator-(T v2, const FTensor<T, shapes...> &v1) {
  return Fixed::element_wise_op<OP::Binary::minus>(v1, v2);
}

template <typename T, size_t... shapes>
BlasForceInline BlasCudaConstruc auto
operator*(T v2, const FTensor<T, shapes...> &v1) {
  return Fixed::element_wise_op<OP::Binary::mul>(v1, v2);
}

template <typename T, size_t... shapes>
BlasForceInline BlasCudaConstruc auto
operator/(T v2, const FTensor<T, shapes...> &v1) {
  return Fixed::element_wise_op<OP::Binary::div>(v1, v2);
}

template <typename T, size_t N>
BlasForceInline BlasCudaConstruc auto normal(const FTensor<T, N> &v1) {
  return 1.0f / sqrt(dot(v1, v1)) * v1;
}

template <typename T, size_t... shapes>
BlasForceInline BlasCudaConstruc auto
lowerBound(const FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  return Fixed::element_wise_op<OP::Binary::_min>(v1, v2);
}

template <typename T, size_t... shapes>
BlasForceInline BlasCudaConstruc auto
upperBound(const FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  return Fixed::element_wise_op<OP::Binary::_max>(v1, v2);
}

} // namespace Expblas