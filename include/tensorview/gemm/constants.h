#pragma once
namespace tv {

namespace gemm {

constexpr int kStrided = 0;
constexpr int kContig = 1;
constexpr int kWarpSize = 32;
constexpr int kOptimAccessBit = 128;
constexpr int kOptimAccessBytes = kOptimAccessBit / 8;

} // namespace gemm
} // namespace tv