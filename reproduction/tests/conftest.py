"""Use the same single-threaded, deterministic runtime as training."""
import os
import sys
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmame_rt.determinism import apply_runtime_flags

apply_runtime_flags()
