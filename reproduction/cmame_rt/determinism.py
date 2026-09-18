"""Runtime determinism flags shared by every training and evaluation process.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

import os
import warnings

from .protocol import runtime_cfg

_RT = runtime_cfg()

CUBLAS_ENV_KEY = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_ENV_VALUE: str = str(_RT["CUBLAS_WORKSPACE_CONFIG"])
FLOAT32_MATMUL_PRECISION: str = str(_RT["float32_matmul_precision"])
TORCH_NUM_THREADS: int = int(_RT["torch_num_threads"])
TORCH_NUM_INTEROP_THREADS: int = int(_RT["torch_num_interop_threads"])


def export_cublas_env() -> bool:
    """Set ``CUBLAS_WORKSPACE_CONFIG`` if absent; True when it was already set."""
    current = os.environ.get(CUBLAS_ENV_KEY)
    if current == CUBLAS_ENV_VALUE:
        return True
    if current is not None:
        warnings.warn("%s was %r, overriding with %r"
                      % (CUBLAS_ENV_KEY, current, CUBLAS_ENV_VALUE),
                      RuntimeWarning, stacklevel=2)
    os.environ[CUBLAS_ENV_KEY] = CUBLAS_ENV_VALUE
    return False


def apply_runtime_flags(verbose: bool = False) -> dict:
    """Apply every protocol.runtime determinism flag to the current process."""
    import torch

    preset = export_cublas_env()
    if not preset:
        warnings.warn(
            "%s was not exported before process start; set it in the worker "
            "environment so CUDA sees it at initialisation" % CUBLAS_ENV_KEY,
            RuntimeWarning, stacklevel=2)

    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = bool(_RT["cudnn_deterministic"])
    torch.backends.cudnn.benchmark = bool(_RT["cudnn_benchmark"])
    torch.use_deterministic_algorithms(bool(_RT["use_deterministic_algorithms"]))
    torch.set_float32_matmul_precision(FLOAT32_MATMUL_PRECISION)
    torch.set_num_threads(TORCH_NUM_THREADS)
    try:
        torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)
    except RuntimeError:
        # Interop thread count is only settable before parallel work starts.
        pass

    state = runtime_state()
    state["cublas_env_preset"] = preset
    if verbose:
        print(state)
    return state


def runtime_state() -> dict:
    """Return the observed determinism-relevant runtime state."""
    import torch

    return {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else None),
        "default_dtype": str(torch.get_default_dtype()),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "num_threads": int(torch.get_num_threads()),
        "num_interop_threads": int(torch.get_num_interop_threads()),
        "cublas_workspace_config": os.environ.get(CUBLAS_ENV_KEY),
    }


def assert_runtime_flags() -> dict:
    """Raise ``AssertionError`` when any determinism flag deviates."""
    import torch

    s = runtime_state()
    problems = []
    if s["default_dtype"] != "torch.float32":
        problems.append("default dtype %s" % s["default_dtype"])
    if s["cuda_matmul_allow_tf32"] or s["cudnn_allow_tf32"]:
        problems.append("TF32 enabled")
    if s["cudnn_deterministic"] is not bool(_RT["cudnn_deterministic"]):
        problems.append("cudnn.deterministic")
    if s["cudnn_benchmark"] is not bool(_RT["cudnn_benchmark"]):
        problems.append("cudnn.benchmark")
    if s["deterministic_algorithms"] is not bool(
            _RT["use_deterministic_algorithms"]):
        problems.append("use_deterministic_algorithms")
    if s["float32_matmul_precision"] != FLOAT32_MATMUL_PRECISION:
        problems.append("float32_matmul_precision %s"
                        % s["float32_matmul_precision"])
    if s["num_threads"] != TORCH_NUM_THREADS:
        problems.append("num_threads %s" % s["num_threads"])
    if s["cublas_workspace_config"] != CUBLAS_ENV_VALUE:
        problems.append("%s=%r" % (CUBLAS_ENV_KEY, s["cublas_workspace_config"]))
    if torch.cuda.is_available() and not s["cuda_available"]:  # pragma: no cover
        problems.append("cuda visibility")
    if problems:
        raise AssertionError("runtime flags deviate: " + "; ".join(problems))
    return s


def worker_env(extra: dict = None) -> dict:
    """Environment mapping a spawned worker must inherit."""
    env = dict(os.environ)
    env[CUBLAS_ENV_KEY] = CUBLAS_ENV_VALUE
    if extra:
        env.update(extra)
    return env


__all__ = ["CUBLAS_ENV_KEY", "CUBLAS_ENV_VALUE", "export_cublas_env",
           "apply_runtime_flags", "assert_runtime_flags", "runtime_state",
           "worker_env"]
