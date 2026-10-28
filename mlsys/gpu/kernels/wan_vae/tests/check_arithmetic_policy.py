"""CPU checks for policy selection, benchmark isolation and truthful reporting."""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench_wan_vae_decoder as decoder
import bench_wan_vae_encoder as encoder
import torch
from bench_utils import (
    check_tf32_model,
    compare_model_outputs,
    configure_arithmetic,
    strict_accuracy_reference,
    validation_status,
    validation_succeeded,
)
from mega_wan_vae._policy import check_model_policy, model_tf32_policy


def main():
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    try:
        for benchmark in (encoder, decoder):
            for flags, allowed in (([], False), (["--allow-tf32"], True), (["--no-allow-tf32"], False)):
                with patch.object(sys, "argv", ["benchmark", "--dtype", "fp32", *flags]):
                    args = benchmark.parse_args()
                assert args.allow_tf32 is allowed
                assert not args.accuracy_diagnostics
                with redirect_stdout(io.StringIO()) as output:
                    configure_arithmetic(args)
                assert torch.backends.cudnn.allow_tf32 is allowed
                assert not torch.backends.cuda.matmul.allow_tf32
                assert "Accuracy diagnostics:" not in output.getvalue()
                assert model_tf32_policy(torch.float32, None) is allowed
                check_model_policy(allowed)
                try:
                    check_model_policy(not allowed)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Changed framework policy was accepted")
            with patch.object(sys, "argv", ["benchmark", "--dtype", "fp32", "--allow-tf32", "--accuracy-diagnostics"]):
                args = benchmark.parse_args()
            assert args.accuracy_diagnostics
            with redirect_stdout(io.StringIO()) as output:
                configure_arithmetic(args)
            assert "Accuracy diagnostics: extra strict FP32" in output.getvalue()

        args = SimpleNamespace(dtype="fp32", allow_tf32=True, accuracy_diagnostics=False)
        with redirect_stdout(io.StringIO()) as output:
            configure_arithmetic(args, model_validation=False)
        assert "Diffusers" not in output.getvalue(), "Primitive benchmark claimed a model reference"
        torch.backends.cudnn.allow_tf32 = True

        with patch.object(torch.cuda, "synchronize") as synchronize:
            for dtype, allow_tf32, diagnostics in (
                ("fp32", True, False),
                ("fp32", False, True),
                ("bf16", False, True),
            ):
                skipped = SimpleNamespace(dtype=dtype, allow_tf32=allow_tf32, accuracy_diagnostics=diagnostics)
                with patch("builtins.print") as emit, patch("torch.ones") as call:
                    assert strict_accuracy_reference(skipped, call) is None
                    call.assert_not_called()
                    emit.assert_not_called()
            synchronize.assert_not_called()
        args.accuracy_diagnostics = True

        def reference():
            assert not torch.backends.cudnn.allow_tf32
            return torch.ones(1)

        with patch.object(torch.cuda, "synchronize"), redirect_stdout(io.StringIO()):
            assert strict_accuracy_reference(args, reference).item() == 1
        assert torch.backends.cudnn.allow_tf32
        try:
            strict_accuracy_reference(args, lambda: (_ for _ in ()).throw(RuntimeError("reference failure")))
        except RuntimeError:
            pass
        assert torch.backends.cudnn.allow_tf32, "Accuracy failure leaked policy"
        assert validation_status([], args) != "ALL PASS"
        assert validation_succeeded(validation_status([], args))
        assert not validation_succeeded(validation_status(["Mega FAIL"], args))
        assert not validation_succeeded("OOM")
        reference = torch.ones(1, 4, 1, 2, 2)
        emit = lambda line: None
        assert check_tf32_model("encoder", reference * 1.009, reference, operation="encode", emit=emit)
        assert not check_tf32_model("encoder", reference * 1.011, reference, operation="encode", emit=emit)
        # A large logvar scale must not hide bad posterior means.
        reference[:, 2:] = 1000
        candidate = reference.clone()
        candidate[:, :2] += 0.02
        assert not check_tf32_model("encoder", candidate, reference, operation="encode", emit=emit)
        reference = torch.zeros(1, 3, 1, 2, 2)
        candidate = reference + 0.009
        lines = []
        assert compare_model_outputs(
            "TF32", candidate, reference, atol=1e-4, rtol=1e-4, allow_tf32=True, operation="decode", emit=lines.append
        )
        assert len(lines) == 1 and lines[0].startswith("  PASS TF32 (TF32 model criterion)")
        lines.clear()
        assert compare_model_outputs(
            "TF32",
            candidate,
            reference,
            atol=1e-4,
            rtol=1e-4,
            allow_tf32=True,
            operation="decode",
            accuracy_diagnostics=True,
            emit=lines.append,
        )
        assert "DIAG outside tolerance" in lines[0]
        assert any("PASS TF32 (TF32 model criterion)" in line for line in lines)
        assert not any("FAIL" in line for line in lines)
        assert not compare_model_outputs(
            "IEEE", candidate, reference, atol=1e-4, rtol=1e-4, operation="decode", emit=emit
        )
        assert not check_tf32_model("decoder", reference + 0.011, reference, operation="decode", emit=emit)
        assert not check_tf32_model("decoder", reference + float("nan"), reference, operation="decode", emit=emit)
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved
    print("Arithmetic CLI, model policy, isolated reference and status checks PASS")


if __name__ == "__main__":
    main()
