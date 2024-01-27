#!/bin/bash

# alias
alias lt='ls -lhrt'

# utils
infoMsg() {
    echo -e "\033[32;1m${1}\033[0m"
}

warnMsg() {
    echo -e "\033[31;1m${1}\033[0m"
}

checkMsg() {
    echo -e "\033[33;4m${1}\033[0m"
}

checkMsg "Before ..."
echo $CPATH
echo $LIBRARY_PATH
echo $LD_LIBRARY_PATH

# boost
if ! [[ "${LD_LIBRARY_PATH}" =~ "boost" ]]; then
    CM_SHARED_BOOST_PATH="/cm/shared/apps/boost/current"
    CM_LOCAL_BOOST_PATH="/cm/local/apps/boost/current"
    if [ -d $CM_SHARED_BOOST_PATH ]; then
        CM_BOOST_PATH=$CM_SHARED_BOOST_PATH
    elif [ -d $CM_LOCAL_BOOST_PATH ]; then
        CM_BOOST_PATH=$CM_LOCAL_BOOST_PATH
    else
        warnMsg "boost not found in shared place; try 'module load boost'?"
    fi
    if [ -n "${CM_BOOST_PATH}" ]; then
        infoMsg "Found boost at ${CM_BOOST_PATH}/"
        # for cmake to use boost
        export BOOST_ROOT="${CM_BOOST_PATH}"
        export BOOST_INCLUDEDIR="${CM_BOOST_PATH}/include"
        export BOOST_LIBRARYDIR="${CM_BOOST_PATH}/lib64"
        # system paths
        export CPATH="${CM_BOOST_PATH}/include:${CPATH}"
        export LIBRARY_PATH="${CM_BOOST_PATH}/lib64:${LIBRARY_PATH}"
        export LD_LIBRARY_PATH="${CM_BOOST_PATH}/lib64:${LD_LIBRARY_PATH}"
    fi
fi

# cuda
if ! [[ "${LD_LIBRARY_PATH}" =~ "cm/shared/apps/cuda" ]]; then
    CM_CUDA_PATH="/cm/shared/apps/cuda-latest/toolkit/current"
    if [ -d $CM_CUDA_PATH ]; then
        infoMsg "Found cuda at ${CM_CUDA_PATH}/"
        export CUDA_PATH="${CM_CUDA_PATH}"
        export PATH="${CM_CUDA_PATH}/bin:${PATH}"
        export CPATH="${CM_CUDA_PATH}/include:${CPATH}"
        export LIBRARY_PATH="${CM_CUDA_PATH}/lib64:${LIBRARY_PATH}"
        export LD_LIBRARY_PATH="${CM_CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
    else
        warnMsg "cuda not found in shared place"
    fi
fi
# for cmake?
if [[ "${CUDA_HOME}" != "${CUDA_PATH}" ]]; then
    infoMsg "Correct CUDA_HOME from ${CUDA_HOME}"
    export CUDA_HOME="${CUDA_PATH}"
fi
# for cmake
if [ -z "${CUDACXX}" ]; then
    infoMsg "set CUDACXX"
    export CUDACXX="${CUDA_PATH}/bin/nvcc"
fi

# openmpi
if [[ "${LD_LIBRARY_PATH}" =~ "cm/shared/apps/openmpi" ]]; then
    warnMsg "If openmpi under /cm/shared/ not working, you can 'module unload openmpi'"
fi

checkMsg "After ..."
echo $CPATH
echo $LIBRARY_PATH
echo $LD_LIBRARY_PATH
