#pragma once
#include "dispatch_cpu.h"
#ifdef USE_CUDA
#include "dispatch_gpu.cu"
#endif
#include "dot_dispatch.h"

namespace Expblas {
template <typename Saver, typename Container, typename DataType, typename Exp,
          OperatorType exp_type, typename... Args>
inline void Execute(TensorBase<Container, DataType, Device::CPU> *dst,
                    const ExpBase<Exp, DataType, exp_type> &exp,
                    Args &&...args) {
  auto dst_shape = ShapeCheck::Check(dst->derived_to());
  auto exp_shape = ShapeCheck::Check(exp.derived_to());

  if (dst_shape == exp_shape) {
    CPUEngine<Saver, TensorBase<Container, DataType, Device::CPU>, Exp,
              exp_type>::dispatch(dst, exp, std::forward<Args>(args)...);
  }
}

template <Device device> struct ReduceTemplate;

template <> struct ReduceTemplate<Device::CPU> {
  template <typename Saver, typename Reducer, typename T, typename Exp,
            OperatorType exp_type>
  static auto eval(const ExpBase<Exp, T, exp_type> &exp, int axis) {
    auto exp_shape = ShapeCheck::Check(exp.derived_to());
    auto dst_shape = exp_shape;
    dst_shape[axis] = 1;
    auto res = Tensor<T, Meta::traits<decltype(exp_shape)>::dim,
                      Meta::traits<Exp>::device>(dst_shape);
    allocSpace(res);
    if (axis == Meta::traits<decltype(exp_shape)>::dim - 1)
      nonPacketReduceLdimExecute<Saver, Reducer>(
          &res, MakeExpWrapper(exp.derived_to()), exp_shape);
    else
      nonPacketReduceHdimExecute<Saver, Reducer>(
          &res, MakeExpWrapper(exp.derived_to()), exp_shape, axis);
    return res;
  }
};

#ifdef USE_CUDA
template <typename Saver, typename DataType, int Dim, typename Exp,
          OperatorType exp_type, typename... Args>
inline void Execute(Tensor<DataType, Dim, Device::GPU> *dst,
                    const ExpBase<Exp, DataType, exp_type> &exp,
                    Args &&...args) {
  Shape<Dim> dst_shape = ShapeCheck::Check(dst->derived_to());
  auto exp_shape = ShapeCheck::Check(exp.derived_to());

  if (dst_shape == exp_shape) {
    GPUEngine<Saver, Tensor<DataType, Dim, Device::GPU>, Exp,
              exp_type>::dispatch(dst, exp, std::forward<Args>(args)...);
  }
}

template <> struct ReduceTemplate<Device::GPU> {
  template <typename Saver, typename Reducer, typename T, typename Exp,
            OperatorType exp_type>
  static auto eval(const ExpBase<Exp, T, exp_type> &exp, int axis) {
    auto exp_shape = ShapeCheck::Check(exp.derived_to());
    auto dst_shape = exp_shape;
    dst_shape[axis] = 1;
    auto res = Tensor<T, Meta::traits<decltype(exp_shape)>::dim, Device::GPU>(
        dst_shape);
    allocSpace(res);
    if (axis == Meta::traits<decltype(exp_shape)>::dim - 1)
      GPUReduceLdimExecute<Saver, Reducer>(
          &res, MakeExpWrapper(exp.derived_to()), exp_shape);
    else
      GPUReduceHdimExecute<Saver, Reducer>(
          &res, MakeExpWrapper(exp.derived_to()), exp_shape, axis);
    return res;
  }
};
#endif // USE_CUDA

template <typename Saver, typename DataType> struct ExpEngine {
  template <typename Dst, typename Exp, OperatorType exp_type, typename... Args>
  inline static void eval(Dst *dst, const ExpBase<Exp, DataType, exp_type> &exp,
                          Args &&...args) {
    Execute<Saver>(dst, exp, std::forward<Args>(args)...);
  }
  template <typename Dst, typename Exp>
  inline static void
  eval(Dst *dst, const ExpBase<Exp, DataType, OperatorType::advance> &exp) {
    AdvanceEngine<DataType>::eval(dst, exp.derived_to());
  }
};

} // namespace Expblas