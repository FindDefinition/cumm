#pragma once
namespace tv {

namespace gemm {
namespace constants{

constexpr int kStridedAxis = 0;
constexpr int kContigAxis = 1;
constexpr int kWarpSize = 32;
constexpr int kOptimAccessBit = 128;
constexpr int kOptimAccess = kOptimAccessBit / 8;

}

} // namespace gemm
} // namespace tv