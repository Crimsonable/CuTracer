#pragma once
#include "base.h"

namespace Expblas {
template <typename Container, typename T, Device device> class TensorBase;
template <typename T, int Dim, Device device> class Tensor;
template <typename T, size_t... shapes> class FTensor;
template <typename ElementType> class Scalar;

template <typename Derived, typename DataType, OperatorType type> class ExpBase;
template <typename Op, typename Lhs, typename Rhs, typename DataType,
          OperatorType type>
class BinaryExp;
template <typename Op, typename Exp, typename DataType, OperatorType exp_type>
class UnaryExp;
template <typename Exp, typename DataType> class TransposeExp;
template <typename LTensor, typename RTensor, typename DataType, bool transL,
          bool transR>
class DotExp;

template <typename Saver, typename DataType> struct ExpEngine;
template <typename DataType> struct AdvanceEngine;
template <typename Op, typename Dst, typename Exp, OperatorType exp_type>
struct CPUEngine;
template <typename Op, typename Dst, typename Exp, OperatorType exp_type>
struct GPUEngine;
template <typename Exp> struct ExpWrapper;
template <typename Exp, Arch arch> struct PacketExpWrapper;

template <int N> struct Shape;
struct ShapeCheck;
template <typename Exp> struct DimCheck;

namespace Meta {
template <typename Derived, typename DType, OperatorType type>
struct traits<ExpBase<Derived, DType, type>> {
  using DataType = DType;
  constexpr static auto device = traits<Derived>::device;
};

template <int N> struct traits<Shape<N>> { constexpr static int dim = N; };

template <typename T, int Dim, Device _device>
struct traits<Tensor<T, Dim, _device>> {
  using DataType = T;
  constexpr static int dim = Dim;
  constexpr static auto device = _device;
};

template <typename T, size_t... shapes> struct traits<FTensor<T, shapes...>> {
  using DataType = T;
  constexpr static int dim = sizeof...(shapes);
  constexpr static auto device = Device::CPU;
};

template <typename Op, typename Lhs, typename Rhs, typename DataType,
          OperatorType type>
struct traits<BinaryExp<Op, Lhs, Rhs, DataType, type>> {
  constexpr static int dim =
      std::max(Meta::traits<Lhs>::dim, Meta::traits<Rhs>::dim);
  constexpr static int lhs_dim = Meta::traits<Lhs>::dim;
  constexpr static int rhs_dim = Meta::traits<Rhs>::dim;
  constexpr static auto device = traits<Lhs>::device;
};

template <typename Op, typename Exp, typename DataType, OperatorType exp_type>
struct traits<UnaryExp<Op, Exp, DataType, exp_type>> {
  constexpr static int dim = Meta::traits<Exp>::dim;
  constexpr static auto device = traits<Exp>::device;
};

template <typename Exp, typename DataType>
struct traits<TransposeExp<Exp, DataType>> {
  constexpr static int dim = Meta::traits<Exp>::dim;
  constexpr static auto device = traits<Exp>::device;
};

template <typename Exp, Arch _arch>
struct traits<PacketExpWrapper<Exp, _arch>> {
  constexpr static auto arch = _arch;
};
} // namespace Meta

} // namespace Expblas