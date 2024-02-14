#!/usr/bin/env python3

import pprint
import pycuda.driver as cuda


def CompCap2Throughput(major, minor):
    # Define the mapping of compute capability to FLOP per SM
    # (ver_major, ver_minor) : (fp32, fp64, int32, tensor_core count, tc_flops)
    throughput_mapping = {
        (3, 0): (192, 0, 0, 0, 0),
        (3, 2): (192, 0, 0, 0, 0),
        (3, 5): (192, 0, 0, 0, 0),
        (3, 7): (192, 0, 0, 0, 0),
        (5, 0): (128, 0, 0, 0, 0),
        (5, 2): (128, 0, 0, 0, 0),
        (5, 3): (128, 0, 0, 0, 0),
        (6, 0): (64, 0, 0, 0, 0),
        (6, 1): (128, 0, 0, 0, 0),
        (6, 2): (128, 0, 0, 0, 0),
        # 8 mixed-precision Tensor Cores for deep learning matrix arithmetic
        # https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727/2
        (7, 0): (64, 32, 64, 8, 1024),
        (7, 2): (64, 32, 64, 8, 1024),
        (7, 5): (64, 32, 64, 8, 1024),  # Turing
        # 4 mixed-precision Third-Generation Tensor Cores supporting half-precision (fp16), __nv_bfloat16, tf32, sub-byte and double precision (fp64) matrix arithmetic for compute capabilities 8.0, 8.6 and 8.7
        (8, 0): (64, 32, 64, 4, 2048),  # A100, A30
        (8, 6): (128, 2, 64, 4, 2048),  # A40
        (8, 7): (128, 2, 64, 4, 2048),
        # 4 mixed-precision Fourth-Generation Tensor Cores supporting fp8, fp16, __nv_bfloat16, tf32, sub-byte and fp64 for compute capability 8.9
        (8, 9): (128, 2, 64, 4, 2048),  # L40, Ada family
        # 4 mixed-precision Fourth-generation Tensor Cores supporting the new FP8 input type in either E4M3 or E5M2 for exponent (E) and mantissa (M), half-precision (fp16), __nv_bfloat16, tf32, INT8 and double precision (fp64) matrix arithmetic
        (9, 0): (128, 64, 64, 4, 4096),
    }
    # check if we know the mapping
    if (major, minor) in throughput_mapping:
        return throughput_mapping[(major, minor)]
    else:
        return (0, 0, 0, 0, 0)


def get_device_properties(device_idx=0, verbose=0):
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
    # Ada  https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
    # A100 https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
    # H100 https://resources.nvidia.com/en-us-tensor-core

    cuda.init()
    device_count = cuda.Device.count()
    if device_count < 1:
        return {}

    for i in range(device_count):
        device = cuda.Device(i)
        properties = device.get_attributes()
        if verbose >= 2:
            print(f"Device {i} ({device.name()}) Properties:")
            pprint.pprint(properties)

    properties = cuda.Device(i).get_attributes()
    sm_clock = properties[cuda.device_attribute.CLOCK_RATE]
    sm_count = properties[cuda.device_attribute.MULTIPROCESSOR_COUNT]
    ver_major = properties[cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR]
    ver_minor = properties[cuda.device_attribute.COMPUTE_CAPABILITY_MINOR]
    fp32_to_fp64_ratio = properties[
        cuda.device_attribute.SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO
    ]
    (
        fp32_per_sm,
        fp64_per_sm,
        int32_per_sm,
        tensorcore_per_sm,
        tc_flops_per_clk_per_sm,
    ) = CompCap2Throughput(ver_major, ver_minor)
    fp32_core_count = fp32_per_sm * sm_count
    tensor_core_count = tensorcore_per_sm * sm_count
    fp32_flops = fp32_per_sm * sm_count * sm_clock * 2 / 1e9
    fp16_flops = fp32_flops * 2
    fp64_flops = fp32_flops / fp32_to_fp64_ratio
    fp16tc_flops = tc_flops_per_clk_per_sm * sm_count * sm_clock / 1e9
    memory_clock = properties[cuda.device_attribute.MEMORY_CLOCK_RATE]
    memory_bus_width = properties[cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]
    memory_bandwidth = memory_clock * memory_bus_width * 2 / 8e9
    tc_flop_per_byte = fp16tc_flops / memory_bandwidth
    if verbose >= 1:
        print("=========================================")
        print(f"SMs: {sm_count}")
        print(f"fp32 Cores: {fp32_core_count}")
        print(f"Tensor Cores: {tensor_core_count}")
        print(f"SM Clock: {sm_clock/ 1e6:.2f} GHz")
        print(f"fp64: {fp64_flops:.2f} TFLOPs")
        print(f"fp32: {fp32_flops:.2f} TFLOPs")
        print(f"fp16: {fp16_flops:.2f} TFLOPs")
        print(f"int8: {fp32_flops * 4:.2f} TFLOPs")
        print(f"tc-fp16: {fp16tc_flops :.2f} TFLOPs")
        print(f"memory bw: {memory_bandwidth :.2f} TB/s")
        print(f"TC flop_per_byte: {tc_flop_per_byte :.2f}")
        print("=========================================")

    return {
        "sm_count": sm_count,
        "fp32_core_count": fp32_core_count,
        "tensor_core_count": tensor_core_count,
        "sm_clock": sm_clock,
        "fp64_flops": fp64_flops,
        "fp32_flops": fp32_flops,
        "fp16_flops": fp16_flops,
        "tc_fp16_flops": fp16tc_flops,
        "memory_clock": memory_clock,
        "memory_bus_width": memory_bus_width,
        "memory_bandwidth": memory_bandwidth,
        "tc_flop_per_byte": tc_flop_per_byte,
    }


def print_flops(tflops: float, op_type: str, prefix="run"):
    props = get_device_properties()
    perc = tflops * 100 / props[op_type]
    print(f"{prefix}: {tflops:.2f} TFLOPs ({perc:.0f}%)", flush=True)


if __name__ == "__main__":
    get_device_properties(verbose=2)
