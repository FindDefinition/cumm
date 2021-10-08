#pragma once
#include <tensorview/core/defs.h>
#include <tensorview/core/all.h>
namespace tv {
namespace gemm {
namespace layout {

using index_t = int32_t;
using long_index_t = int64_t;

template <int Interleave> struct ColumnMajorInterleaved;
template <int Interleave> struct RowMajorInterleaved {
  /* for a interleaved [4, 4] rowmajor matrix, all contiguous coord is
     0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3
     0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3
  */
  static constexpr int kInterleave = Interleave;
  using transposed_t = ColumnMajorInterleaved<kInterleave>;

  index_t stride;
  TV_HOST_DEVICE_INLINE constexpr RowMajorInterleaved(index_t stride_)
      : stride(stride_) {}
  TV_HOST_DEVICE_INLINE constexpr static RowMajorInterleaved
  from_shape(const tv::array<int, 2> &shape) {
    return RowMajorInterleaved(shape[1] * Interleave);
  }

  TV_HOST_DEVICE_INLINE constexpr long_index_t operator()(index_t x,
                                                          index_t y) const {
    index_t row_major = x / Interleave;
    index_t row_minor = x % Interleave;
    return long_index_t(row_major) * long_index_t(stride) +
           long_index_t(y) * Interleave + row_minor;
  }
  template <int Axis>
  TV_HOST_DEVICE_INLINE constexpr index_t inverse(long_index_t offset) const {
    if (Axis == 0) {
      index_t row_major = index_t(offset / stride);
      index_t residual = index_t(offset % stride);
      index_t row_minor = residual % Interleave;
      return row_major * Interleave + row_minor;
    } else {
      return (offset % stride) / Interleave;
    }
  }
};

template <int Interleave> struct ColumnMajorInterleaved {

  static constexpr int kInterleave = Interleave;
  using transposed_t = RowMajorInterleaved<kInterleave>;

  index_t stride;
  TV_HOST_DEVICE_INLINE constexpr ColumnMajorInterleaved(index_t stride_)
      : stride(stride_) {}
  TV_HOST_DEVICE_INLINE constexpr static ColumnMajorInterleaved
  from_shape(const tv::array<int, 2> &shape) {
    return ColumnMajorInterleaved(shape[0] * Interleave);
  }

  TV_HOST_DEVICE_INLINE constexpr long_index_t operator()(index_t x,
                                                          index_t y) const {
    index_t column_major = y / Interleave;
    index_t column_minor = y % Interleave;
    return long_index_t(column_major) * long_index_t(stride) +
           long_index_t(x) * Interleave + column_minor;
  }
  template <int Axis>
  TV_HOST_DEVICE_INLINE constexpr index_t inverse(long_index_t offset) const {
    if (Axis == 1) {
      index_t column_major = index_t(offset / stride);
      index_t residual = index_t(offset % stride);
      index_t column_minor = residual % Interleave;
      return column_major * Interleave + column_minor;
    } else {
      return (offset % stride) / Interleave;
    }
  }
};

template <>
struct RowMajorInterleaved<1> {
  index_t stride;

  using transposed_t = ColumnMajorInterleaved<1>;
  static constexpr int kInterleave = 1;

  TV_HOST_DEVICE_INLINE constexpr RowMajorInterleaved(index_t stride_) : stride(stride_) {}
  TV_HOST_DEVICE_INLINE constexpr static RowMajorInterleaved
  from_shape(const tv::array<int, 2> &shape) {
    return RowMajorInterleaved(shape[1]);
  }
  TV_HOST_DEVICE_INLINE constexpr long_index_t operator()(index_t x,
                                                          index_t y) const {
    return long_index_t(x) * long_index_t(stride) + y;
  }
  template <int Axis>
  TV_HOST_DEVICE_INLINE constexpr index_t inverse(long_index_t offset) const {
    if (Axis == 0) {
      return index_t(offset / stride);
    } else {
      return index_t(offset % stride);
    }
  }
};

template <>
struct ColumnMajorInterleaved<1>  {
  index_t stride;
  using transposed_t = RowMajorInterleaved<1>;
  static constexpr int kInterleave = 1;

  TV_HOST_DEVICE_INLINE constexpr ColumnMajorInterleaved(index_t stride_)
      : stride(stride_) {}
  TV_HOST_DEVICE_INLINE constexpr static ColumnMajorInterleaved
  from_shape(const tv::array<int, 2> &shape) {
    return ColumnMajorInterleaved(shape[0]);
  }

  TV_HOST_DEVICE_INLINE constexpr long_index_t operator()(index_t x,
                                                          index_t y) const {
    return long_index_t(y) * long_index_t(stride) + x;
  }
  template <int Axis>
  TV_HOST_DEVICE_INLINE constexpr index_t inverse(long_index_t offset) const {
    if (Axis == 0) {
      return index_t(offset % stride);
    } else {
      return index_t(offset / stride);
    }
  }
};


using RowMajor = RowMajorInterleaved<1>;
using ColumnMajor = ColumnMajorInterleaved<1>;



} // namespace layout

} // namespace gemm
} // namespace tv