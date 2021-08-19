#pragma once
#include <utility>

namespace Expblas {
namespace Meta {
template <typename T> struct traits;

template <typename T1, typename T2> struct TypeResolution;

struct TrueTag {};
struct FalseTag {};
template <bool Flag> struct TagDispatch;
template <> struct TagDispatch<true> { using type = TrueTag; };
template <> struct TagDispatch<false> { using type = FalseTag; };

/*template <typename... Args> constexpr auto FoldMul(Args... args) {
  return (... * args);
}*/

template <typename Head, typename... Tail>
BlasCudaConstruc constexpr auto FoldMul_naive(Head head, Tail... tail) {
  return head * FoldMul_naive(tail...);
}

template <typename Head>
BlasCudaConstruc constexpr auto FoldMul_naive(Head head) {
  return head;
}

template <int idx> struct getI;
template <int idx> struct getI {
  template <typename Head, typename... Tail>
  BlasCudaConstruc constexpr static auto get(Head head, Tail... tail) {
    return getI<idx - 1>::get(tail...);
  }
};

template <> struct getI<0> {
  template <typename Head, typename... Tail>
  BlasCudaConstruc constexpr static auto get(Head head, Tail... tail) {
    return head;
  }
};

template <typename... Args>
BlasCudaConstruc constexpr auto get_last(Args... args) {
  return getI<sizeof...(Args) - 1>::get(args...);
}

template <size_t N, typename Func, typename... Args>
BlasCudaConstruc void LoopUnroll(Func &&f, Args &&...args) {
  f(N, std::forward<Args>(args)...);
  if constexpr (N)
    LoopUnroll<N - 1>(std::forward<Func>(f), std::forward<Args>(args)...);
}

template <int N, typename T, typename... Args>
BlasCudaConstruc void init_helper(T *data, Args &&...args) {
  data[N] = getI<N>::get(std::forward<Args>(args)...);
  if constexpr (N)
    init_helper<N - 1>(data, std::forward<Args>(args)...);
}

template <typename T, typename... Args>
BlasCudaConstruc void array_init(T *data, Args &&...args) {
  init_helper<sizeof...(args) - 1>(data, std::forward<Args>(args)...);
}

/* template <typename T, typename... Args, size_t... N>
BlasCudaConstruc void init_helper(T *data, Args &&...args,
                              std::index_sequence<N...>) {
  ((data[N] = args), ...);
}

template <typename T, typename... Args>
BlasCudaConstruc void array_init(T *data, Args &&...args) {
  init_helper<T, Args...>(data, std::forward<Args>(args)...,
                          std::make_index_sequence<sizeof...(args)>{});
}*/

} // namespace Meta
} // namespace Expblas