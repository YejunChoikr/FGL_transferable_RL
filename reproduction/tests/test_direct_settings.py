"""Direct-optimization integration tests.

These are implementation checks, not experiments: the time cap must be refused
when it is not frozen, every optimizer setting must come from the protocol and
be readable back out of the library, and the default objective must remain the
canonical reward of ``spec/reference_math.py``. S10 selects its separate
bilateral objective through the common runner.

Run: ``python -m pytest tests/test_direct_settings.py`` from the project root,
or ``python tests/test_direct_settings.py``.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "spec"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reference_math  # noqa: E402

from cmame_rt import direct  # noqa: E402

PROTOCOL = json.loads((ROOT / "spec" / "protocol.json").read_text(encoding="utf-8"))
CFG = PROTOCOL["direct_optimization"]


def test_methods_match_protocol() -> None:
    assert list(direct.METHODS) == list(CFG["methods"])
    assert direct.N_VARS == int(CFG["variables"])
    assert CFG["bounds"] == [-1.0, 1.0]
    assert CFG["domain"] == "upper"


def test_ga_mutation_prob_and_prob_var_are_distinct() -> None:
    ga = CFG["GA"]
    assert ga["mutation_individual_prob"] == 1.0
    assert abs(ga["mutation_variable_prob"] - 1.0 / 30.0) < 1e-15
    assert ga["mutation_individual_prob"] != ga["mutation_variable_prob"]


def test_time_cap_is_refused_when_not_frozen() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "locks").mkdir()
        try:
            direct.load_time_cap(root)
        except direct.TimeCapMissing:
            pass
        else:
            raise AssertionError("a missing DIRECT_TIME_CAP.json must refuse the run")
        (root / "locks" / "DIRECT_TIME_CAP.json").write_text(
            json.dumps({"time_cap_seconds": 12.5}), encoding="utf-8")
        cap = direct.load_time_cap(root)
        assert cap["time_cap_seconds"] == 12.5
        assert len(cap["sha256"]) == 64


def test_time_cap_rejects_nonpositive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "locks").mkdir()
        (root / "locks" / "DIRECT_TIME_CAP.json").write_text(
            json.dumps({"time_cap_seconds": 0.0}), encoding="utf-8")
        try:
            direct.load_time_cap(root)
        except direct.TimeCapMissing:
            return
        raise AssertionError("a non-positive cap must be refused")


def test_subseed_follows_protocol_rule() -> None:
    for seed in (0, 1, 2, 3, 4):
        expected = int(np.random.SeedSequence(
            [20260913, seed, 1700]).generate_state(1, dtype=np.uint32)[0])
        assert direct.subseed(seed) == expected
    assert direct.DIRECT_SEARCH_STREAM == PROTOCOL["runtime"]["rng_stream_ids"]["direct_search"]


def test_objective_matches_reference_math() -> None:
    from objective_fixture import build_pilot_objective, check_against_reference

    import torch

    objective, n_params = build_pilot_objective(goal=5, device=torch.device("cpu"))
    assert n_params == PROTOCOL["surrogate"]["architecture"]["expected_trainable_params"]
    agreement = check_against_reference(objective)
    assert agreement["max_encode_abs_diff"] == 0.0
    # float32 surrogate against the float64 reference; the reward numerator is a
    # difference of millimetre-scale displacements, so single-precision
    # cancellation sets the floor here.
    assert agreement["max_reward_abs_diff"] < 1e-5


def test_objective_gradient_matches_finite_difference() -> None:
    from objective_fixture import build_pilot_objective

    import torch

    objective, _ = build_pilot_objective(goal=5, device=torch.device("cpu"))
    rng = np.random.default_rng(3)
    x = rng.uniform(-0.9, 0.9, size=30)
    value, grad, u9 = objective.objective_and_grad(x)
    assert u9.shape == (9,)
    step = 1e-3
    for index in (0, 7, 29):
        plus = x.copy(); plus[index] += step
        minus = x.copy(); minus[index] -= step
        numeric = (objective.objective(plus)[0] - objective.objective(minus)[0]) / (2 * step)
        assert abs(numeric - grad[index]) < 5e-3 * max(1.0, abs(grad[index]))


def test_each_method_runs_and_respects_the_cap() -> None:
    from objective_fixture import build_pilot_objective

    import torch

    cap = {"time_cap_seconds": 3.0, "file": "test", "sha256": "0" * 64, "payload": {}}
    tmp = Path(tempfile.mkdtemp())
    try:
        for method in direct.METHODS:
            case = {"case_id": f"test/{method}", "kind": "optimizer", "method": method,
                    "seed": 0, "domain": "upper", "goals": [5], "C1": 0.5, "C2": 1.0}
            objective, _ = build_pilot_objective(goal=5, device=torch.device("cpu"))
            result = direct.run_direct(case, tmp / method, torch.device("cpu"),
                                       objective=objective,
                                       surrogate_source="test-random-cnn",
                                       time_cap=cap)
            assert result["complete"], method
            assert not result["nonfinite"], method
            counts = json.loads((tmp / method / "counts.json").read_text(encoding="utf-8"))
            timing = json.loads((tmp / method / "timing.json").read_text(encoding="utf-8"))
            incumbent = json.loads((tmp / method / "incumbent.json").read_text(encoding="utf-8"))
            resolved = json.loads((tmp / method / "resolved_config.json").read_text(
                encoding="utf-8"))
            assert resolved["library_operators_readback"]["match"], method
            assert resolved["rng_stream_id"] == 1700
            assert counts["candidates_within_cap"] + counts["overrun_candidates"] \
                   == counts["candidates_total"], method
            assert counts["candidates_within_cap"] > 0, method
            assert not direct._missing_or_null(timing, direct.REQUIRED_TIMING_KEYS), method
            assert not direct._missing_or_null(counts, direct.REQUIRED_COUNTS_KEYS), method
            assert timing["n_candidates_within_cap"] == counts["candidates_within_cap"]
            assert timing["time_cap_s"] == cap["time_cap_seconds"], method
            assert abs(timing["t_search_s"] + timing["overrun_s"]
                       - timing["t_total_s"]) < 1e-6, method
            assert timing["t_total_s"] >= cap["time_cap_seconds"] - 0.5, method
            assert timing["t_load_s"] >= 0.0, method
            assert isinstance(counts["batch_eval_time_s"], float), method
            assert timing["exclusive"] is False, method
            assert result["exclusive"] is False, method
            assert len(incumbent["actions30"]) == 30
            assert len(incumbent["physical_thickness30"]) == 30
            assert incumbent["thickness_bounds_mm"] == [1.2, 2.4]
            thickness = np.asarray(incumbent["physical_thickness30"])
            assert thickness.min() >= 1.2 - 1e-9 and thickness.max() <= 2.4 + 1e-9
            expected = reference_math.physical_thickness(
                np.asarray(incumbent["actions30"]), 1.2, 2.4)
            assert np.max(np.abs(thickness - expected)) < 1e-12
            for name in ("search_log.csv", "resolved_config.json", "incumbent.json",
                         "counts.json", "timing.json", "result.json"):
                assert (tmp / method / name).is_file(), (method, name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scaler_accepts_tensors_ndarrays_and_lists() -> None:
    """Regression: load_surrogate_for_env returns CUDA tensors.

    The first production batch of ten direct-optimization cases failed instantly
    with ``TypeError: can't convert cuda:0 device type tensor to numpy`` because
    build_objective wrapped the scaler in ``np.asarray``. Every conversion on the
    loader boundary now goes through as_device_tensor / as_numpy.
    """
    import torch

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    for device in devices:
        forms = {
            "cuda_tensor": torch.arange(9, dtype=torch.float32,
                                        device="cuda:0" if torch.cuda.is_available()
                                        else "cpu") + 1.0,
            "cpu_tensor": torch.arange(9, dtype=torch.float64) + 1.0,
            "ndarray": np.arange(9, dtype=np.float64) + 1.0,
            "list": [float(i + 1) for i in range(9)],
        }
        for name, value in forms.items():
            tensor = direct.as_device_tensor(value, device)
            assert tuple(tensor.shape) == (9,), (name, device)
            assert tensor.dtype == torch.float32, (name, device)
            assert tensor.device.type == device.type, (name, device)
            array = direct.as_numpy(value)
            assert array.shape == (9,) and array.dtype == np.float64, (name, device)
        for scaler in ({"mean": forms["cuda_tensor"], "scale": forms["cuda_tensor"]},
                       {"mean": forms["list"], "scale": forms["list"]}):
            assert direct.as_device_tensor(
                direct._scaler_field(scaler, "mean"), device).shape == (9,)


def test_objective_accepts_tensor_and_list_designs() -> None:
    from objective_fixture import build_pilot_objective

    import torch

    objective, _ = build_pilot_objective(goal=5, device=torch.device("cpu"))
    base = np.full(30, 0.1)
    forms = [base.tolist(), base, torch.as_tensor(base, dtype=torch.float64)]
    values = [objective.rewards(f)[0][0] for f in forms]
    assert max(abs(v - values[0]) for v in values) == 0.0
    for form in forms:
        value, grad, u9 = objective.objective_and_grad(form)
        assert grad.shape == (30,) and u9.shape == (9,)
        assert np.isfinite(value)


def test_no_naive_numpy_conversion_on_the_loader_boundary() -> None:
    """Static guard: np.asarray must not be applied to loader-supplied objects."""
    source = (ROOT / "cmame_rt" / "direct.py").read_text(encoding="utf-8")
    for forbidden in ('np.asarray(scaler[', 'np.asarray(scaler.',
                      'np.array(scaler[', '.numpy()' ):
        if forbidden == '.numpy()':
            # allowed only behind an explicit .detach().cpu()
            for line in source.splitlines():
                if '.numpy()' in line:
                    assert '.detach().cpu().numpy()' in line, line
            continue
        assert forbidden not in source, forbidden


def test_run_direct_accepts_the_run_case_contract() -> None:
    """run_case calls run_direct(case, attempt_dir, device, exclusive=, adam_backend=)."""
    import inspect

    params = inspect.signature(direct.run_direct).parameters
    assert list(params)[:3] == ["case", "attempt_dir", "device"]
    for name in ("exclusive", "adam_backend"):
        assert name in params, name
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def test_exclusive_reaches_timing_from_kwarg_and_from_the_case() -> None:
    from objective_fixture import build_pilot_objective

    import torch

    cap = {"time_cap_seconds": 2.0, "file": "test", "sha256": "0" * 64, "payload": {}}
    tmp = Path(tempfile.mkdtemp())
    try:
        for label, case_extra, kwargs in (
            ("kwarg", {}, {"exclusive": True}),
            ("case_field", {"_exclusive": True}, {}),
            ("default", {}, {}),
        ):
            case = {"case_id": "test/exclusive/" + label, "kind": "optimizer",
                    "method": "DE", "seed": 0, "domain": "upper", "goals": [5],
                    "C1": 0.5, "C2": 1.0}
            case.update(case_extra)
            objective, _ = build_pilot_objective(goal=5, device=torch.device("cpu"))
            result = direct.run_direct(case, tmp / label, torch.device("cpu"),
                                       objective=objective, time_cap=cap,
                                       surrogate_source="test", **kwargs)
            timing = json.loads((tmp / label / "timing.json").read_text(encoding="utf-8"))
            expected = label != "default"
            assert timing["exclusive"] is expected, (label, timing["exclusive"])
            assert result["exclusive"] is expected, label
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_unknown_keyword_arguments_are_tolerated() -> None:
    """run_case may grow the contract; an extra keyword must not fail the run."""
    from objective_fixture import build_pilot_objective

    import torch

    cap = {"time_cap_seconds": 2.0, "file": "test", "sha256": "0" * 64, "payload": {}}
    tmp = Path(tempfile.mkdtemp())
    try:
        case = {"case_id": "test/extra_kwarg", "kind": "optimizer", "method": "DE",
                "seed": 0, "domain": "upper", "goals": [5], "C1": 0.5, "C2": 1.0}
        objective, _ = build_pilot_objective(goal=5, device=torch.device("cpu"))
        result = direct.run_direct(case, tmp / "DE", torch.device("cpu"),
                                   objective=objective, time_cap=cap,
                                   surrogate_source="test",
                                   some_future_kwarg=123)
        assert result["complete"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_required_field_fails_the_attempt() -> None:
    """A null required field must produce complete=false, never a silent gap."""
    from objective_fixture import build_pilot_objective

    import torch

    cap = {"time_cap_seconds": 2.0, "file": "test", "sha256": "0" * 64, "payload": {}}
    tmp = Path(tempfile.mkdtemp())
    original = direct.REQUIRED_TIMING_KEYS
    try:
        direct.REQUIRED_TIMING_KEYS = original + ("a_field_nobody_writes",)
        case = {"case_id": "test/selfcheck", "kind": "optimizer", "method": "DE",
                "seed": 0, "domain": "upper", "goals": [5], "C1": 0.5, "C2": 1.0}
        objective, _ = build_pilot_objective(goal=5, device=torch.device("cpu"))
        result = direct.run_direct(case, tmp / "DE", torch.device("cpu"),
                                   objective=objective, time_cap=cap,
                                   surrogate_source="test")
        assert result["complete"] is False
        assert result["error_type"] == "IncompleteTimingRecord"
        assert "a_field_nobody_writes" in result["missing_timing_keys"]
    finally:
        direct.REQUIRED_TIMING_KEYS = original
        shutil.rmtree(tmp, ignore_errors=True)


def test_incumbent_is_the_best_candidate_inside_the_cap() -> None:
    """Candidates logged after the cap must never win the incumbent."""
    log = direct.SearchLog.__new__(direct.SearchLog)
    log.rows = [
        {"within_cap": True, "reward": 1.0, "index": 0},
        {"within_cap": False, "reward": 9.0, "index": 1},
        {"within_cap": True, "reward": 2.0, "index": 2},
    ]
    log.designs = [np.zeros(30), np.ones(30), np.full(30, 0.5)]
    log.u9 = [np.zeros(9)] * 3
    best = direct.SearchLog.incumbent(log)
    assert best["index"] == 2 and best["canonical_reward"] == 2.0


def test_bilateral_log_keeps_eq4_reporting_separate() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        log = direct.SearchLog(tmp / "search.csv", 0.0, float("inf"), 5,
                               "bilateral")
        actions = np.zeros((1, 30))
        u9 = np.array([[0., 1., 2., 4., 8., 7., 3., 2., 1.]])
        log.add_batch("BO", actions, np.array([1.0]), u9, 0.01)
        log.close()
        best = log.incumbent()
        assert best["bilateral_reward"] == 1.0
        assert best["training_objective"] == 1.0
        assert best["canonical_reward"] == 2.5
        row = json.loads(json.dumps(best))
        assert row["canonical_reward"] != row["bilateral_reward"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _main() -> int:
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("failures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
