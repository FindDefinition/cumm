#pragma once
#include <tensorview/core/all.h>
#include <tensorview/core/defs.h>
namespace tv {
namespace gemm {
namespace layout {

using index_t = int32_t;
using long_index_t = int64_t;

template <int ElementSize, int KBlock> struct VoltaTensorOpCrosswise {
  // aggregate initialization produce wrong result in cicc stage (only affect static assert, ptx code is correct)
  // currently a template wrapper looks ok.
  static constexpr auto kTileShape = mp_array_int_v<8, 4>;      // with epa
  static constexpr auto kPartitionShape = mp_array_int_v<4, 4>; // with epa
  static constexpr int kAccessSize = 64;

  static constexpr int kElementsPerAccess = kAccessSize / ElementSize;
  index_t stride;
  TV_HOST_DEVICE_INLINE constexpr VoltaTensorOpCrosswise(index_t stride_)
      : stride(stride_) {}
  TV_HOST_DEVICE_INLINE constexpr static VoltaTensorOpCrosswise
  from_shape(const tv::array<int, 2> &shape) {
    return VoltaTensorOpCrosswise(shape[1]);
  }

  TV_HOST_DEVICE_INLINE constexpr long_index_t operator()(index_t x,
                                                          index_t y) const {
    int vec_contiguous_idx = y / kElementsPerAccess;
    int vec_strided_idx = x;
    int vec_strided_within_tile = vec_contiguous_idx & 0x7;
    // 0: tile: 4x64, a smem bank
    // 1. map to tile offset. assume we have 4x128, so tile offset
    // is 0, 64, 128, 192, ...
    int permuted_vec_contiguous  =  vec_strided_idx & (~0xF);
    // 2. inside a tile, map to each permuted sub tile 4x16
    // (0,4,8,12)[], (0,16,32,48)[]
    permuted_vec_contiguous += (vec_strided_idx & 0x3) * 4;
    permuted_vec_contiguous += (((vec_strided_idx >> 2) ^ ((vec_strided_idx & 0x10) >> 3)) & 0x3);
    // 3. generate permuted offset
    permuted_vec_contiguous ^= ((vec_strided_within_tile >> 1) & 0x3);

    int permuted_vec_strided = vec_contiguous_idx;
    int element_contiguous = permuted_vec_contiguous *  kElementsPerAccess + 
                             (y % kElementsPerAccess);
  
    return element_contiguous + permuted_vec_strided * (stride * kElementsPerAccess);
  }
};

template <bool OperandA, int ElementSize> struct VoltaTensorOpCongruous {
  static constexpr auto kTileShape = mp_array_int_v<4, 8>;      // with epa
  static constexpr auto kPartitionShape = mp_array_int_v<OperandA ? 4 : 2, OperandA ? 4 : 8>; // with epa

  static constexpr int kAccessSize = 128;
  static constexpr int kElementsPerAccess = kAccessSize / ElementSize;
  index_t stride;
  TV_HOST_DEVICE_INLINE constexpr VoltaTensorOpCongruous(index_t stride_)
      : stride(stride_) {}
  TV_HOST_DEVICE_INLINE constexpr static VoltaTensorOpCongruous
  from_shape(const tv::array<int, 2> &shape) {
    return VoltaTensorOpCongruous(shape[1]);
  }
  TV_HOST_DEVICE_INLINE constexpr long_index_t operator()(index_t x,
                                                          index_t y) const {
    int vec_contiguous_idx = y / kElementsPerAccess;
    int vec_strided_idx = x;

    // Compute the fundamental tile being accessed
    int tile_contiguous_idx = vec_contiguous_idx / kTileShape[1];
    int tile_strided_idx = vec_strided_idx / kTileShape[0];

    int tile_contiguous_residual = vec_contiguous_idx % kTileShape[1];
    int tile_strided_residual = vec_strided_idx % kTileShape[0];

    int permuted_strided_within_tile;
    int permuted_contiguous_within_tile;
    if TV_IF_CONSTEXPR (OperandA) {
      permuted_strided_within_tile = (tile_contiguous_residual >> 1);
      permuted_contiguous_within_tile =
          (tile_strided_residual ^ permuted_strided_within_tile) |
          ((tile_contiguous_residual & 1) << 2);

    } else {
      permuted_strided_within_tile = (tile_contiguous_residual & 0x3);
      permuted_contiguous_within_tile =
          (tile_strided_residual ^ permuted_strided_within_tile) |
          (tile_contiguous_residual & 0x4);
    }

    // Compute final element location
    int element_contiguous = (tile_contiguous_idx * kTileShape[1] +
                              permuted_contiguous_within_tile) *
                                 kElementsPerAccess +
                             (y % kElementsPerAccess);

    int element_strided =
        tile_strided_idx * kTileShape[0] + permuted_strided_within_tile;

    auto res = element_contiguous + element_strided * stride;
    // tv::printf2_block_once(threadIdx.x, stride_,
    // "VoltaTensorOpMultiplicandBCongruous", res, coord.strided(),
    // coord.contiguous());
    return res;
  }
};

} // namespace layout

} // namespace gemm
} // namespace tv