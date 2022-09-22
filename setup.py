#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import shutil
import sys
from pathlib import Path
from shutil import rmtree
from typing import List

import pccm
from pccm.extension import ExtCallback, PCCMBuild, PCCMExtension
from setuptools import Command, find_packages, setup
from setuptools.extension import Extension

# Package meta-data.
NAME = 'cumm'
RELEASE_NAME = NAME
cuda_ver = os.getenv("CUMM_CUDA_VERSION", None)

if cuda_ver is not None and cuda_ver != "":
    cuda_ver = cuda_ver.replace(".", "")  # 10.2 to 102
    RELEASE_NAME += "-cu{}".format(cuda_ver)

DESCRIPTION = 'CUda Matrix Multiply library'
URL = 'https://github.com/FindDefinition/cumm'
EMAIL = 'yanyan.sub@outlook.com'
AUTHOR = 'Yan Yan'
REQUIRES_PYTHON = '>=3.6'
VERSION = None

# What packages are required for this module to be executed?
REQUIRED = [
    "pccm<0.4.0", "ccimport<0.4.0", "pybind11>=2.6.0", "fire", "numpy", 
    "contextvars; python_version == \"3.6\"",
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(str(Path(__file__).parent))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open('version.txt', 'r') as f:
        version = f.read().strip()
else:
    version = VERSION
cwd = os.path.dirname(os.path.abspath(__file__))


def _convert_build_number(build_number):
    parts = build_number.split(".")
    if len(parts) == 2:
        return "{}{:03d}".format(int(parts[0]), int(parts[1]))
    elif len(parts) == 1:
        return build_number
    else:
        raise NotImplementedError


env_suffix = os.environ.get("CUMM_VERSION_SUFFIX", "")
if env_suffix != "":
    version += ".dev{}".format(_convert_build_number(env_suffix))
version_path = os.path.join(cwd, NAME, '__version__.py')
about['__version__'] = version

with open(version_path, 'w') as f:
    f.write("__version__ = '{}'\n".format(version))


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds...')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(
            sys.executable))

        self.status('Uploading the package to PyPI via Twine...')
        os.system('twine upload dist/*')

        self.status('Pushing git tags...')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


class CopyHeaderCallback(ExtCallback):
    def __call__(self, ext: Extension, package_dir: Path,
                 built_target_path: Path):
        include_path = package_dir / "cumm" / "include"
        if include_path.exists():
            shutil.rmtree(include_path)
        code_path = Path(__file__).parent / "include"
        shutil.copytree(code_path, include_path)


disable_jit = os.getenv("CUMM_DISABLE_JIT", None)

if disable_jit is not None and disable_jit == "1":
    cmdclass = {
        'upload': UploadCommand,
        'build_ext': PCCMBuild,
    }
    from cumm.csrc.arrayref import ArrayPtr
    from cumm.tensorview_bind import TensorViewBind
    cus = [ArrayPtr(), TensorViewBind()]

    if cuda_ver is None or (cuda_ver is not None and cuda_ver != ""):
        pass
    ext_modules: List[Extension] = [
        PCCMExtension(cus,
                      "cumm/core_cc",
                      Path(__file__).resolve().parent / "cumm",
                      extcallback=CopyHeaderCallback())
    ]
else:
    cmdclass = {
        'upload': UploadCommand,
    }
    ext_modules = []

# Where the magic happens:
setup(
    name=RELEASE_NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests', )),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    entry_points={
        'console_scripts': [],
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
