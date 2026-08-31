import pytest
import torch

from sage_avo.runtime import select_torch_device, torch_runtime_report


def test_runtime_report_contains_required_diagnostics():
    report = torch_runtime_report()
    assert set(report) == {
        "sys.executable",
        "torch.__version__",
        "torch.version.cuda",
        "torch.cuda.is_available()",
        "torch.cuda.get_device_name(0)",
    }


def test_required_cuda_refuses_cpu_fallback():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        select_torch_device("cpu", require_cuda=True, context="test")


def test_explicit_cpu_is_visible():
    with pytest.warns(RuntimeWarning, match="selected CPU"):
        assert select_torch_device("cpu") == torch.device("cpu")
