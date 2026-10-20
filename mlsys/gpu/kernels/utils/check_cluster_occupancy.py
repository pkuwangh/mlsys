#!/usr/bin/env python3
"""Report SM count and CUDA-predicted usable SMs for 2/4/8/16-CTA clusters.

Requires cuda-python and a cluster-capable GPU. A one-CTA-per-SM probe isolates
cluster placement limits; real kernels can have different resource limits.
No kernel is executed. CUDA's occupancy prediction is not a scheduling guarantee.
"""

import argparse

from cuda.bindings import driver as cuda

# Minimal PTX avoids requiring nvcc, NVRTC, or CUTLASS for an occupancy query.
PROBE_PTX = b"""
.version 8.0
.target sm_90
.address_size 64
.visible .entry occupancy_probe() { ret; }
\0
"""


def checked(result):
    """Unpack a CUDA result, raising on unexpected driver errors."""
    status, *values = result
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA: {status.name}")
    return values[0] if values else None


def report(device):
    """Query cluster occupancy in the selected device's primary context."""
    attr = cuda.CUdevice_attribute

    def device_attr(name):
        return checked(cuda.cuDeviceGetAttribute(getattr(attr, f"CU_DEVICE_ATTRIBUTE_{name}"), device))

    name = checked(cuda.cuDeviceGetName(256, device)).split(b"\0", 1)[0].decode()
    sm_count = device_attr("MULTIPROCESSOR_COUNT")
    print(f"{name} | SM count: {sm_count}")
    if not device_attr("CLUSTER_LAUNCH"):
        print("Thread-block clusters are not supported on this device.")
        return

    context = checked(cuda.cuDevicePrimaryCtxRetain(device))
    checked(cuda.cuCtxPushCurrent(context))
    module = None
    try:
        module = checked(cuda.cuModuleLoadData(PROBE_PTX))
        kernel = checked(cuda.cuModuleGetFunction(module, b"occupancy_probe"))
        # More than half of each SM's shared memory prevents two resident CTAs.
        shared_bytes = (device_attr("MAX_SHARED_MEMORY_PER_MULTIPROCESSOR") // 2048 + 1) * 1024
        func_attr = cuda.CUfunction_attribute
        checked(
            cuda.cuFuncSetAttribute(kernel, func_attr.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes)
        )
        checked(cuda.cuFuncSetAttribute(kernel, func_attr.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1))
        blocks = checked(cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(kernel, 128, shared_bytes))
        if blocks != 1:
            raise RuntimeError(f"Expected one resident CTA per SM, got {blocks}")

        print(f"Probe: 128 threads/CTA, {shared_bytes // 1024} KiB shared/CTA, one resident CTA/SM")
        print("Cluster CTAs | Active clusters | Usable SMs | Unused SMs")
        print("-------------+-----------------+------------+-----------")
        for size in (2, 4, 8, 16):
            cluster = cuda.CUlaunchAttribute()
            cluster.id = cuda.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            cluster.value.clusterDim.x = size
            cluster.value.clusterDim.y = cluster.value.clusterDim.z = 1
            config = cuda.CUlaunchConfig()
            config.gridDimX = ((sm_count + size - 1) // size) * size
            config.gridDimY = config.gridDimZ = 1
            config.blockDimX = 128
            config.blockDimY = config.blockDimZ = 1
            config.sharedMemBytes = shared_bytes
            config.hStream = cuda.CUstream(0)
            config.attrs = [cluster]
            config.numAttrs = 1
            result = cuda.cuOccupancyMaxActiveClusters(kernel, config)
            if result[0] == cuda.CUresult.CUDA_ERROR_INVALID_CLUSTER_SIZE:
                print(f"{size:12} | unsupported")
                continue
            active = checked(result)
            usable = active * size
            print(f"{size:12} | {active:15} | {usable:10} | {sm_count - usable:10}")
        print("16-CTA clusters use the nonportable opt-in; counts are driver predictions.")
    finally:
        if module is not None:
            checked(cuda.cuModuleUnload(module))
        checked(cuda.cuCtxPopCurrent())
        checked(cuda.cuDevicePrimaryCtxRelease(device))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=int, default=0, help="CUDA-visible device ordinal")
    args = parser.parse_args()
    checked(cuda.cuInit(0))
    report(checked(cuda.cuDeviceGet(args.device)))


if __name__ == "__main__":
    main()
