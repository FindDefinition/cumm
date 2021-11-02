# cumm
CUda Matrix Multiply library.

[![Build Status](https://github.com/FindDefinition/cumm/workflows/build/badge.svg)](https://github.com/FindDefinition/cumm/actions?query=workflow%3Abuild)

```cumm``` is developed during learning of [CUTLASS](https://github.com/NVIDIA/cutlass), which use too much c++ template and make code unmaintainable. So I develop [pccm](https://github.com/FindDefinition/PCCM), use python as meta programming language, to replace c++ template meta programming. 
Now ```pccm``` become a foundational framework of ```cumm``` and my other c++ project such as [spconv](https://github.com/traveller59/spconv). 
```cumm``` also contains a python asyncio-based gemm simulator that **share same meta program** with CUDA code, enable gemm visualization and easy debug experience.

## Install

### Prebuilt

We offer python 3.7-3.10 and cuda 10.2/11.1/11.3/11.4 prebuilt binaries for linux (manylinux).

We offer python 3.7-3.10 and cuda 10.2/11.1/11.3/11.4 prebuilt binaries for windows 10/11.

We will offer prebuilts for CUDA versions supported by latest pytorch release. For example, pytorch 1.9 support cuda 10.2 and 11.1, so we support them too.

```pip install cumm-cu102``` for CUDA 10.2

```pip install cumm-cu111``` for CUDA 11.1

```pip install cumm-cu113``` for CUDA 11.3

```pip install cumm-cu114``` for CUDA 11.4

### Build from source

#### Linux

1. install build-essential, install CUDA
2. run ```export CUMM_DISABLE_JIT="1"```
3. run ```python setup.py install```/```pip install -e .```/```python setup.py bdist_wheel```+```pip install dists/xxx.whl```

#### Windows 10/11

1. install visual studio 2019 or newer. make sure C++ development package is installed. install CUDA
2. set [powershell script execution policy](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.1)
3. start a new powershell, run ```tools/msvc_setup.ps1```
4. run ```$Env:CUMM_DISABLE_JIT = "1"```
5. run ```python setup.py install```/```pip install -e .```/```python setup.py bdist_wheel```+```pip install dists/xxx.whl```

## Note
The work is done when the author is an employee at Tusimple.