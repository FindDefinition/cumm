#pragma once
#include <iterator>
#include <tensorview/core/all.h>
#include <tensorview/gemm/core/all.h>
#include <tensorview/gemm/math/all.h>
#include <type_traits>

namespace tv {

namespace gemm {

template <
    typename TOut, int Count, typename TAcc = TOut, typename TCompute = TOut,
    typename UnaryOp = math::UnaryIdentity<TCompute, size_t(Count)>,
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
struct LinearCombination {
  using ElementOutput = TOut;
  using ElementAccumulator = TAcc;
  using ElementCompute = TCompute;

  static constexpr int kCount = Count;

  using FragmentOutput = tv::array<ElementOutput, kCount>;
  using FragmentAccumulator = tv::array<ElementAccumulator, kCount>;
  using FragmentCompute = tv::array<ElementCompute, kCount>;

  ElementCompute alpha, beta;
  TV_HOST_DEVICE_INLINE constexpr LinearCombination(
      ElementCompute alpha_ = ElementCompute(1),
      ElementCompute beta_ = ElementCompute(0))
      : alpha(alpha_), beta(beta_) {}
  TV_HOST_DEVICE_INLINE constexpr bool is_source_needed() const {
    return beta != ElementCompute(0);
  }
  TV_HOST_DEVICE_INLINE FragmentOutput
  operator()(FragmentAccumulator const &accumulator,
             FragmentOutput const &source) const {
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round>
        source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;
    FragmentCompute converted_source = source_converter(source);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    FragmentCompute intermediate;

    math::multiplies<FragmentCompute> mul_add_source;
    math::multiply_add<FragmentCompute> mul_add_accumulator;

    intermediate =
        mul_add_source(beta, converted_source); // X =  beta * C + uniform
    intermediate = mul_add_accumulator(alpha, converted_accumulator,
                                       intermediate); // D = alpha * Accum + X
    UnaryOp op;
    intermediate = op(intermediate);
    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    return destination_converter(intermediate);
  }
  TV_HOST_DEVICE_INLINE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    FragmentCompute converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    FragmentCompute intermediate;
    math::multiplies<FragmentCompute> mul_accumulator;
    intermediate =
        mul_accumulator(alpha, converted_accumulator); // D = alpha * Accum
    UnaryOp op;
    intermediate = op(intermediate);
    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;
    return destination_converter(intermediate);
  }
};

} // namespace gemm
} // namespace tv