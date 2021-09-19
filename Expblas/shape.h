#pragma once
#include "metatools.h"
#include "pre_declearation.h"
#include <initializer_list>

namespace Expblas {
using std::initializer_list;

template <> struct Shape<0> {};

template <int N> struct Shape {
  Shape() {}

  Shape(const initializer_list<size_t> &_list) {
    for (auto i = _list.begin(); i != _list.end(); ++i)
      dims[i - _list.begin()] = *i;
  }

  BlasForceInline BlasCudaFunc size_t &operator[](int idx) { return dims[idx]; }
  BlasForceInline BlasCudaFunc size_t operator[](int idx) const {
    return dims[idx];
  }

  void operator=(const Shape<N> &other) {
    for (int i = 0; i < N; ++i)
      dims[i] = other[i];
  }

  BlasForceInline constexpr size_t size() const {
    size_t count = 1;
    for (int i = 0; i < N; ++i)
      count *= dims[i];
    return count;
  }

  BlasForceInline Shape<N> strides() const {
    Shape<N> res = *this;
    res[N - 1] = 1;
    for (int i = N - 2; i >= 0; --i)
      res[i] *= res[i + 1];
    return res;
  }

  BlasForceInline Shape<2> flat2D() const {
    Shape<2> res;
    res[1] = dims[N - 1];
    res[0] = this->size() / dims[N - 1];
    return res;
  }

  BlasForceInline bool operator==(const Shape<0> &other) const { return true; }

  template <int N1>
  BlasForceInline bool operator==(const Shape<N1> other) const {
    for (int i = 1; i < std::min(N1, N) + 1; ++i) {
      if (other[N1 - i] != dims[N - i])
        return false;
    }
    return true;
  }

  BlasForceInline Shape<N> reverse() const {
    Shape<N> res;
    for (int i = 0; i < N; ++i)
      res[N - i - 1] = dims[i];
    return res;
  }

  BlasForceInline size_t last() const { return dims[N - 1]; }

  BlasForceInline Shape<2> reduce(int axis) {
    Shape<2> res;
    size_t counter = 1;
    for (int i = 0; i <= axis; ++i)
      counter *= dims[i];
    res[0] = counter;
    counter = 1;
    for (int i = axis + 1; i < N; ++i)
      counter *= dims[i];
    res[1] = counter;
    return res;
  }

  size_t dims[N]{0};
};

template <int N1, int N2> auto ShapeMatch(Shape<N1> s1, Shape<N2> s2) {
  constexpr int maxN = std::max(N1, N2);
  constexpr int minN = std::min(N1, N2);
  Shape<maxN> res = N1 > N2 ? s1 : s2;
  for (int i = 1; i < minN + 1; ++i) {
    if ((s1[N1 - i] == 1 || s2[N2 - i] == 1) || s1[N1 - i] == s2[N2 - i])
      res[maxN - i] = std::max(s1[N1 - i], s2[N2 - i]);
    else {
      std::cout << "operands could not be broadcast together" << std::endl;
      abort();
    }
  }
  return res;
}

template <int N1> auto ShapeMatch(Shape<N1> s1, Shape<0> s2) { return s1; }
template <int N2> auto ShapeMatch(Shape<0> s1, Shape<N2> s2) { return s2; }

struct ShapeCheck {
  template <typename T> static Shape<0> Check(const Scalar<T> &scalar) {
    return Shape<0>();
  }

  template <typename Container, typename DataType, Device device>
  static auto Check(const TensorBase<Container, DataType, device> &exp) {
    return exp.get_shape();
  }

  template <typename Op, typename Lhs, typename Rhs, typename DataType,
            OperatorType exp_type>
  static auto Check(const BinaryExp<Op, Lhs, Rhs, DataType, exp_type> &exp) {
    auto lhs_shape = ShapeCheck::Check(exp.lhs.derived_to());
    auto rhs_shape = ShapeCheck::Check(exp.rhs.derived_to());
    return ShapeMatch(lhs_shape, rhs_shape);
  }

  template <typename Op, typename Exp, typename DataType, OperatorType exp_type>
  static auto Check(const UnaryExp<Op, Exp, DataType, exp_type> &exp) {
    return ShapeCheck::Check(exp.self.derived_to());
  }

  template <typename Exp, typename DataType>
  static auto Check(const TransposeExp<Exp, DataType> &exp) {
    auto exp_shape = ShapeCheck::Check(exp.self);
    return exp_shape.reverse();
  }
};

} // namespace Expblas