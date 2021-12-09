# Changelog

## [0.2.8] - 2021-12-8
### Fixed
* Fix missing sm37 in supported arch


## [0.2.7] - 2021-12-8
### Added
* add sm37 for cu102.
* add compile info (cuda arch) for better error information.

## [0.2.6] - 2021-12-3
### Fixed
* Fix a small bug that incorrectly limit arch of simt to sm52.

## [0.2.4] - 2021-11-28
### Added
* add cpu support for CUDAKernelTimer.
* add non-contiguous support for tv::Tensor.
* add tsl hash map, refine cuda hash impl.
### Changed
* raise error instead of exit program when cuda error occurs.
* gemm kernel now use stride, this enable us perform gemm with non-contiguous tensor
### Fixed
* Fix bugs for gemm kernel when use non-contiguous operand.

## [0.2.3] - 2021-11-11
### Fixed
* Fix bugs for implicit gemm

## [0.2.2] - 2021-11-8
### Added
* add support for python 3.6, but cudasim don't support python 3.6.
* add profile tool for all gemm and conv kernels.

## [0.2.1] - 2021-11-8
### Fixed
* Fix some bug of implicit gemm

## [0.2.0] - 2021-11-2
### Addad
* add implicit gemm algorithm for all kind of convolution with kernel volume <= 32. this algorithm is very fast with float16.
* add cuda 11.3 build

### Removed
* remove python 3.6 support