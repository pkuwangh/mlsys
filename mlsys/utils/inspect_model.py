#!/usr/bin/env python3

import argparse
import collections
import types


PROMPT = "Michael Burry is"
MAX_NEW_TOKENS = 128
TAIL_TENSOR_COUNT = 8
MXFP4_TENSOR_ORDER = (
    "gate_up_proj",
    "gate_up_proj_bias",
    "gate_up_proj_precision_config.weight_scale",
    "down_proj",
    "down_proj_bias",
    "down_proj_precision_config.weight_scale",
)


def _format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def _storage_data(value):
    storage = getattr(value, "storage", None)
    return getattr(storage, "data", None)


def _tensor_shape(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        storage_data = _storage_data(value)
        shape = getattr(storage_data, "shape", None)
    if shape is None:
        return None
    return tuple(shape)


def _tensor_dtype(value):
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        storage_data = _storage_data(value)
        dtype = getattr(storage_data, "dtype", None)
    if dtype is None:
        return "unknown"
    return _format_dtype(dtype)


def _format_dtype(dtype):
    dtype = str(dtype).replace("torch.", "")
    if "FloatType(" not in dtype:
        return dtype

    fields = {}
    body = dtype.split("FloatType(", 1)[1].split(")", 1)[0]
    for item in body.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key.strip()] = value.strip()

    exponent = fields.get("bitwidth_exponent")
    mantissa = fields.get("bitwidth_mantissa")
    is_signed = fields.get("is_signed") == "True"
    if exponent is not None and mantissa is not None:
        prefix = "" if is_signed else "u"
        return f"{prefix}e{exponent}m{mantissa}*"

    return dtype


def _tensor_numel(value):
    storage_data = _storage_data(value)
    if hasattr(storage_data, "numel"):
        return storage_data.numel()

    if hasattr(value, "numel"):
        return value.numel()

    shape = _tensor_shape(value)
    if shape is None:
        return 0
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _tensor_element_size(value):
    storage_data = _storage_data(value)
    if hasattr(storage_data, "element_size"):
        return storage_data.element_size()

    if hasattr(value, "element_size"):
        return value.element_size()

    dtype = _tensor_dtype(value)
    if dtype in ("float64", "int64"):
        return 8
    if dtype in ("float32", "int32", "uint32"):
        return 4
    if dtype in ("bfloat16", "float16", "int16", "uint16"):
        return 2
    if dtype in ("uint8", "int8", "bool"):
        return 1
    return 0


def _tensor_nbytes(value):
    return _tensor_numel(value) * _tensor_element_size(value)


def _is_tensor_like(value):
    return _tensor_shape(value) is not None and _tensor_dtype(value) != "unknown"


def _format_shape(value):
    shape = _tensor_shape(value)
    if shape is None:
        return "unknown"
    if len(shape) == 0:
        return "scalar"
    return "x".join(str(dim) for dim in shape)


def _format_local_tensor_name(name):
    return name.replace("_precision_config.weight_scale", "_scale")


def _get_dotted_attr(obj, name):
    value = obj
    for part in name.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _iter_module_tensors(module):
    seen = set()

    if module.__class__.__name__ == "Mxfp4GptOssExperts":
        for name in MXFP4_TENSOR_ORDER:
            value = _get_dotted_attr(module, name)
            if value is None or not _is_tensor_like(value):
                continue
            if "." not in name and name in module._parameters:
                kind = "param"
            elif "." not in name and name in module._buffers:
                kind = "buffer"
            elif name.endswith("weight_scale"):
                kind = "scale"
            else:
                kind = "attr"
            seen.add(name)
            yield name, value, kind

    for name, tensor in module.named_parameters(recurse=False):
        if name in seen:
            continue
        seen.add(name)
        yield name, tensor, "param"

    for name, tensor in module.named_buffers(recurse=False):
        if name in seen:
            continue
        seen.add(name)
        yield name, tensor, "buffer"


def _iter_named_tensors(model):
    for _, tensor, _ in _iter_named_tensors_with_kind(model):
        yield tensor


def _iter_named_tensors_with_kind(model):
    for module_name, module in model.named_modules():
        for name, tensor, kind in _iter_module_tensors(module):
            if module_name:
                name = f"{module_name}.{name}"
            yield name, tensor, kind


def _tensor_summary(model):
    stats = collections.defaultdict(lambda: {"count": 0, "numel": 0, "bytes": 0})
    total_numel = 0
    total_bytes = 0

    for tensor in _iter_named_tensors(model):
        key = _tensor_dtype(tensor)
        numel = _tensor_numel(tensor)
        num_bytes = _tensor_nbytes(tensor)
        stats[key]["count"] += 1
        stats[key]["numel"] += numel
        stats[key]["bytes"] += num_bytes
        total_numel += numel
        total_bytes += num_bytes

    return stats, total_numel, total_bytes


def _print_model_summary(model):
    param_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters()
                          if parameter.requires_grad)
    stats, total_numel, total_bytes = _tensor_summary(model)

    print("Model summary:")
    print(f"  class: {model.__class__.__name__}")
    if hasattr(model, "dtype"):
        print(f"  model dtype: {model.dtype}")
    print(f"  parameters: {param_count:,}")
    print(f"  trainable parameters: {trainable_count:,}")
    print(f"  tensors: {total_numel:,} elements, {_format_bytes(total_bytes)}")
    print("  tensor storage by dtype:")
    for dtype, item in sorted(stats.items()):
        print(
            f"    {dtype:<12} "
            f"{item['count']:>6} tensors "
            f"{item['numel']:>16,} elements "
            f"{_format_bytes(item['bytes']):>12}")


def _print_tensor_table(model, limit):
    tensors = list(_iter_named_tensors_with_kind(model))
    total = len(tensors)
    leading_count = min(max(limit, 0), total)
    tail_start = max(leading_count, total - TAIL_TENSOR_COUNT)
    skipped = tail_start - leading_count

    print(f"Total {total} tensors:")
    _print_tensor_header()
    _print_tensor_rows(tensors[:leading_count])

    if skipped > 0:
        print(f"  ... {skipped} more tensors")

    _print_tensor_rows(tensors[tail_start:])


def _print_tensor_header():
    print(
        f"  {'name':<80} {'kind':<6} {'dtype':<12} "
        f"{'shape':<20} {'size':>12}")


def _print_tensor_rows(tensors):
    for name, tensor, kind in tensors:
        dtype = _tensor_dtype(tensor)
        print(
            f"  {name:<80} {kind:<6} {dtype:<12} "
            f"{_format_shape(tensor):<20} {_format_bytes(_tensor_nbytes(tensor)):>12}")


def _add_dtype_to_repr(model):
    for module in model.modules():
        old_extra_repr = module.extra_repr

        def extra_repr_with_dtype(self, old_extra_repr=old_extra_repr):
            base = old_extra_repr()
            local_tensors = list(_iter_module_tensors(self))
            tensors = [tensor for _, tensor, _ in local_tensors]
            if not tensors:
                return base

            dtypes = sorted({_tensor_dtype(tensor) for tensor in tensors})
            tensor_reprs = [
                f"{_format_local_tensor_name(name)}[{_format_shape(tensor)}]"
                for name, tensor, _ in local_tensors
            ]
            suffix = f"dtype={'+'.join(dtypes)}"
            if tensor_reprs:
                suffix = f"{suffix}, tensors={', '.join(tensor_reprs)}"
            return f"{base}, {suffix}" if base else suffix

        module.extra_repr = types.MethodType(extra_repr_with_dtype, module)


def _load_model(args):
    from transformers import AutoModelForCausalLM

    load_kwargs = {
        "dtype": "auto",
        "local_files_only": True,
    }
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    try:
        return AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except TypeError as exc:
        if "dtype" not in load_kwargs:
            raise
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        try:
            return AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        except TypeError:
            raise exc


def _run_inference(args, model, torch):
    from transformers import AutoTokenizer

    if args.gpu:
        model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    inputs = tokenizer(PROMPT, return_tensors="pt")
    inputs = inputs.to(model.device)

    print("Inputs:")
    for name, value in inputs.items():
        print(f"  {name}: shape={tuple(value.shape)}, dtype={value.dtype}")

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def main(args):
    # Lazy import so the CLI help message can show up quickly.
    import torch

    model = _load_model(args)
    _add_dtype_to_repr(model)

    _print_model_summary(model)
    _print_tensor_table(model, args.parameter_limit)

    print("=============================================")
    print(model)
    print("=============================================")

    if args.run_inference:
        _run_inference(args, model, torch)


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Inspect a Hugging Face causal language model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model name or local path")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow custom model/tokenizer code from the model repo")
    parser.add_argument("--parameter-limit", "-p", type=int, default=24,
                        help=("Number of leading tensors to show; the last "
                              "four tensors are always shown"))
    parser.add_argument("--run-inference", "-r", action="store_true",
                        help="Run a short generation after inspection")
    parser.add_argument("--gpu", action="store_true",
                        help="Move the loaded model to CUDA before inference")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
