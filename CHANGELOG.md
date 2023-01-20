# Changelog

## [0.4.5] - 2023-01-20
### Fixed
- fix a compile problem in msvc

## [0.4.4] - 2023-01-19
### Fixed
- fix unsupported arch in cuda 12.0

## [0.4.3] - 2023-01-19
### Fixed
- fix compile problem

## [0.4.2] - 2023-01-19
### Fixed
- fix some compile problem in cpu only

## [0.4.1] - 2023-01-19
### Changed
- change version to rebuild due to pypi server problem

## [0.4.0] - 2022-12-30
### Added
- Add cuda 12.0
- Add int8 inference for sparse conv 
### Fixed
- Fix some problem in cuda 12.0

## [0.3.7] - 2022-11-05
### Added
- Fix bug in ConvProblem introduced in 0.3.6

## [0.3.6] - 2022-11-05
### Added
- Add int64 support for TensorGeneric

## [0.3.5] - 2022-10-18
### Added
- Add flags for H100 and RTX 4090
### Fixed
- fix nvrtc launch problem when smem size is large
- fix nvrtc constant variable parse problem

## [0.3.4] - 2022-9-25
### Changed
- Change gemm/conv main function to splited version

## [0.3.3] - 2022-9-25
### Fixed
- Fix problem in CompileInfo
### Changed
- Change nlohmann json to 3.11.2

## [0.3.2] - 2022-9-25
### Fixed
- Fix build problem in cuda 10.2
- Fix some bug related to nvrtc

## [0.3.1] - 2022-9-25
### Fixed
- Fix cpu build problem

## [0.3.0] - 2022-9-24
### Added 
- Add Ampere support. faster fp16, faster tf32 and greatly faster int8 kernels in Ampere GPUs.
* Add nvrtc support for conv kernel.
### Removed
- drop python 3.6 support.
### Changed
* BREAKING CHANGE: change dtype enum value for some important reason.

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