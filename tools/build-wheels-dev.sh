#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    outpath="$2"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w "$outpath"
    fi
}

export CUMM_DISABLE_JIT="1"
export CUMM_CUDA_ARCH_LIST="7.5;8.6"
"/opt/python/cp39-cp39/bin/pip" wheel /io/ --no-deps -w /io/wheelhouse_tmp

# Bundle external shared libraries into the wheels
for whl in /io/wheelhouse_tmp/*.whl; do
    repair_wheel "$whl" /io/dist
done

rm -rf /io/wheelhouse_tmp