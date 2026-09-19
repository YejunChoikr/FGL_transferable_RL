"""Direct optimization references BO / L-BFGS-B / GA / DE (owner agent: FEADIR).

Experimental settings are defined in spec/protocol.json.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import platform
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

METHODS = ("BO", "LBFGSB", "GA", "DE")
N_VARS = 30
DIRECT_SEARCH_STREAM = 1700
_SUBSEED_ROOT = 20260913

#: Fields the orchestrator reads out of ``timing.json`` for the workstation
#: timing-coverage report. Every one of them must be non-null or the attempt is
#: failed rather than filed with a silent gap (coordinator, 2026-09-13 15:45).
REQUIRED_TIMING_KEYS = (
    "t_load_s", "t_total_s", "t_search_s", "time_cap_s", "overrun_s",
    "n_candidates_within_cap", "exclusive", "host", "started_at", "finished_at",
)

#: Fixed key names for ``counts.json``.
REQUIRED_COUNTS_KEYS = (
    "surrogate_value_calls", "surrogate_grad_calls", "library_calls",
    "cache_hits", "candidates_total", "candidates_within_cap",
    "batch_eval_time_s", "overrun_candidates",
)


def _sync(device: Any) -> None:
    """Synchronize the CUDA stream so a wall-clock reading is not optimistic."""
    try:
        import torch
    except Exception:  # pragma: no cover
        return
    try:
        dev = torch.device(device)
    except Exception:  # pragma: no cover
        return
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)


def _missing_or_null(payload: dict[str, Any], keys: Sequence[str]) -> list[str]:
    return [k for k in keys if k not in payload or payload[k] is None]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _root() -> Path:
    try:
        from cmame_rt.paths import ROOT  # type: ignore

        return Path(ROOT)
    except Exception:
        return Path(__file__).resolve().parents[1]


def _load_protocol() -> dict[str, Any]:
    try:
        from cmame_rt.protocol import load_protocol  # type: ignore

        return load_protocol()
    except Exception:
        return json.loads((_root() / "spec" / "protocol.json").read_text(encoding="utf-8"))


def subseed(run_seed: int, stream_id: int = DIRECT_SEARCH_STREAM) -> int:
    """Protocol sub-seed rule; delegates to ``cmame_rt.rng`` when it exists."""
    try:
        from cmame_rt.rng import subseed as _subseed  # type: ignore

        return int(_subseed(run_seed, stream_id))
    except Exception:
        seq = np.random.SeedSequence([_SUBSEED_ROOT, int(run_seed), int(stream_id)])
        return int(seq.generate_state(1, dtype=np.uint32)[0])


class TimeCapMissing(RuntimeError):
    """Raised when ``locks/DIRECT_TIME_CAP.json`` is absent or malformed."""


def load_time_cap(root: Path | None = None) -> dict[str, Any]:
    """Read the frozen common time cap. Refuse to run without it."""
    root = root or _root()
    path = root / "locks" / "DIRECT_TIME_CAP.json"
    if not path.is_file():
        raise TimeCapMissing(
            f"{path} is missing. The coordinator freezes one scalar cap (median of the "
            "three upper ST_P0 seed-0 exclusive workstation T_learning values) before "
            "any direct-optimization run; this module will not guess it."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    cap = None
    for key in ("time_cap_seconds", "time_cap_s", "seconds", "cap_seconds"):
        if key in payload:
            cap = float(payload[key])
            break
    if cap is None or not np.isfinite(cap) or cap <= 0:
        raise TimeCapMissing(f"{path} carries no positive time cap: {payload}")
    return {"time_cap_seconds": cap, "time_cap_s": cap, "file": str(path),
            "sha256": _sha256_file(path), "payload": payload}


# --------------------------------------------------------------------------- #
# objective
# --------------------------------------------------------------------------- #
def as_device_tensor(value: Any, device: Any, *, dtype: Any = None) -> Any:
    """Accept a torch tensor (on any device), ndarray, list or scalar.

    ``cmame_rt.surrogate_train.load_surrogate_for_env`` returns the output
    scaler as CUDA tensors, so anything here that went through ``numpy`` first
    raised ``TypeError: can't convert cuda:0 device type tensor to numpy``.
    Converting on the torch side handles every case and never moves a GPU tensor
    through host memory needlessly.
    """
    import torch

    dtype = torch.float32 if dtype is None else dtype
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def as_numpy(value: Any, dtype: Any = np.float64) -> np.ndarray:
    """Accept a torch tensor (on any device), ndarray, list or scalar."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch is always present in production
        torch = None  # type: ignore
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


@dataclass
class SurrogateObjective:
    """Canonical reward of a completed 30-cell design under a frozen surrogate.

    ``model`` maps ``[B, 1, 10, 6]`` to standardized u9; ``y_mean`` / ``y_scale``
    invert the output scaler. The encoding and reward functions are injected so
    that this module never duplicates the SURR-owned definitions.
    """

    model: Any
    y_mean: Any
    y_scale: Any
    goal: int
    C1: float
    C2: float
    device: Any
    encode_batch_torch: Callable[..., Any]
    to_cnn_input: Callable[..., Any]
    reward_torch: Callable[..., Any]
    counts: dict[str, int] = field(default_factory=lambda: {
        "surrogate_value_calls": 0,
        "surrogate_value_candidates": 0,
        "surrogate_grad_calls": 0,
        "surrogate_grad_candidates": 0,
    })

    def _forward(self, actions_t):
        import torch

        x60 = self.encode_batch_torch(
            actions_t, torch.ones_like(actions_t, dtype=torch.bool)
        )
        u9_std = self.model(self.to_cnn_input(x60))
        u9_mm = u9_std * self.y_scale + self.y_mean
        reward = self.reward_torch(u9_mm, actions_t, self.goal, self.C1, self.C2)
        return reward, u9_mm

    def rewards(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Canonical reward and predicted u9 (mm) for a batch of designs."""
        import torch

        batch = np.atleast_2d(as_numpy(actions, np.float32))
        with torch.no_grad():
            actions_t = as_device_tensor(batch, self.device)
            reward, u9 = self._forward(actions_t)
        self.counts["surrogate_value_calls"] += 1
        self.counts["surrogate_value_candidates"] += batch.shape[0]
        return (as_numpy(reward), as_numpy(u9))

    def objective(self, actions: np.ndarray) -> np.ndarray:
        """Return the negative case-selected reward for minimization."""
        reward, _ = self.rewards(actions)
        return -reward

    def objective_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Objective value, its autograd gradient and the predicted u9 (mm)."""
        import torch

        actions_t = as_device_tensor(
            as_numpy(x, np.float32).reshape(1, N_VARS), self.device
        ).requires_grad_(True)
        reward, u9 = self._forward(actions_t)
        loss = -reward.sum()
        grad, = torch.autograd.grad(loss, actions_t)
        self.counts["surrogate_grad_calls"] += 1
        self.counts["surrogate_grad_candidates"] += 1
        return (float(loss.detach().cpu()),
                as_numpy(grad).reshape(N_VARS),
                as_numpy(u9).reshape(9))


def _scaler_field(scaler: Any, name: str) -> Any:
    """Read ``mean`` / ``scale`` from a dict-like or attribute-style scaler."""
    if isinstance(scaler, dict):
        if name in scaler:
            return scaler[name]
        raise KeyError(f"scaler has no {name!r}; keys = {sorted(scaler)}")
    if hasattr(scaler, name):
        return getattr(scaler, name)
    if hasattr(scaler, "__getitem__"):
        return scaler[name]
    raise TypeError(f"cannot read {name!r} from scaler of type {type(scaler).__name__}")


def build_objective(case: dict[str, Any], device: Any) -> tuple[SurrogateObjective, str]:
    """Load the paired transferred surrogate and wire the SURR-owned functions."""
    from cmame_rt import encoding, reward as reward_mod  # type: ignore
    from cmame_rt.surrogate_train import load_surrogate_for_env  # type: ignore
    import torch

    protocol = _load_protocol()
    goal = int(case["goals"][0]) if "goals" in case else int(case["goal"])
    source_case = None
    for dep in case.get("dependencies", []):
        if dep.startswith("surrogate/"):
            source_case = dep
            break
    if source_case is None:
        raise ValueError(f"case {case['case_id']} has no surrogate dependency")
    model, scaler = load_surrogate_for_env(source_case, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # load_surrogate_for_env returns CUDA tensors here; as_device_tensor also
    # accepts ndarray / list / scalar so an alternative loader shape still works.
    y_mean = as_device_tensor(_scaler_field(scaler, "mean"), device)
    y_scale = as_device_tensor(_scaler_field(scaler, "scale"), device)
    for name, vector in (("mean", y_mean), ("scale", y_scale)):
        if tuple(vector.shape) != (9,):
            raise ValueError(f"scaler {name} must have shape (9,), got {tuple(vector.shape)}")
        if not bool(torch.isfinite(vector).all()):
            raise ValueError(f"scaler {name} carries non-finite values")
    if bool((y_scale == 0).any()):
        raise ValueError("scaler scale carries a zero; inverse scaling would be undefined")
    objective_name = str(case.get("objective", "arithmetic"))
    if objective_name == "arithmetic":
        objective_fn = reward_mod.reward_torch
    elif objective_name == "bilateral":
        objective_fn = reward_mod.bilateral_reward_torch
    else:
        raise ValueError("objective must be 'arithmetic' or 'bilateral'")
    return SurrogateObjective(
        model=model, y_mean=y_mean, y_scale=y_scale, goal=goal,
        C1=float(case.get("C1", protocol["reward"]["C1"])),
        C2=float(case.get("C2", protocol["reward"]["C2"])),
        device=device,
        encode_batch_torch=encoding.encode_batch_torch,
        to_cnn_input=encoding.to_cnn_input,
        reward_torch=objective_fn,
    ), source_case


# --------------------------------------------------------------------------- #
# search log
# --------------------------------------------------------------------------- #
class SearchLog:
    """Every evaluated candidate, with the wall clock that decides the cap."""

    FIELDS = ("index", "phase", "batch_index", "batch_size", "elapsed_s",
              "batch_wall_s", "within_cap", "objective",
              "training_objective", "bilateral_reward", "canonical_reward")

    def __init__(self, path: Path, t0: float, deadline: float,
                 goal: int, objective_name: str = "arithmetic"):
        self.path = Path(path)
        self.t0 = t0
        self.deadline = deadline
        self.goal = int(goal)
        self.objective_name = str(objective_name)
        self.index = 0
        self.batch_index = 0
        self.rows: list[dict[str, Any]] = []
        self.designs: list[np.ndarray] = []
        self.u9: list[np.ndarray] = []
        self.n_overrun = 0
        self.batch_eval_time_s = 0.0
        self.last_within_cap_elapsed_s = 0.0
        self._fh = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._fh, fieldnames=list(self.FIELDS) + [f"a{i:02d}" for i in range(N_VARS)]
        )
        self._writer.writeheader()

    def add_batch(self, phase: str, designs: np.ndarray, rewards: np.ndarray,
                  u9: np.ndarray, batch_wall_s: float) -> None:
        finished = time.perf_counter()
        within = finished <= self.deadline
        self.batch_eval_time_s += float(batch_wall_s)
        if within:
            self.last_within_cap_elapsed_s = finished - self.t0
        designs = np.atleast_2d(as_numpy(designs))
        for i in range(designs.shape[0]):
            a = designs[i]
            u = as_numpy(u9[i])
            if self.objective_name == "bilateral":
                j = self.goal - 1
                rho = float(np.mean((a + 1.0) / 2.0))
                canonical = (float(u[j] - 0.5 * (u[j - 1] + u[j + 1]))
                             / (0.5 + rho))
                bilateral = float(rewards[i])
            else:
                canonical = float(rewards[i])
                bilateral = None
            row = {
                "index": self.index, "phase": phase, "batch_index": self.batch_index,
                "batch_size": int(designs.shape[0]),
                "elapsed_s": finished - self.t0, "batch_wall_s": batch_wall_s,
                "within_cap": int(within), "objective": float(-rewards[i]),
                "training_objective": float(rewards[i]),
                "bilateral_reward": bilateral,
                "canonical_reward": canonical,
            }
            for j in range(N_VARS):
                row[f"a{j:02d}"] = float(designs[i, j])
            self._writer.writerow(row)
            self.rows.append({"within_cap": within, "reward": float(rewards[i]),
                              "index": self.index})
            self.designs.append(as_numpy(designs[i]))
            self.u9.append(as_numpy(u9[i]))
            self.index += 1
        if not within:
            self.n_overrun += int(designs.shape[0])
        self.batch_index += 1
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def incumbent(self) -> dict[str, Any] | None:
        valid = [r for r in self.rows if r["within_cap"]]
        if not valid:
            return None
        best = max(valid, key=lambda r: r["reward"])
        k = best["index"]
        objective_name = getattr(self, "objective_name", "arithmetic")
        if objective_name == "arithmetic":
            canonical = best["reward"]
        else:
            a = self.designs[k]
            u = self.u9[k]
            j = self.goal - 1
            rho = float(np.mean((a + 1.0) / 2.0))
            canonical = float(u[j] - 0.5 * (u[j - 1] + u[j + 1])) / (0.5 + rho)
        result = {"index": k, "training_objective": best["reward"],
                  "canonical_reward": canonical,
                  "actions30": self.designs[k].tolist(),
                  "u9_pred_mm": self.u9[k].tolist()}
        if objective_name == "bilateral":
            result["bilateral_reward"] = best["reward"]
        return result


def _evaluate(objective: SurrogateObjective, log: SearchLog, phase: str,
              designs: np.ndarray) -> np.ndarray:
    """Evaluate a batch, log it, and return the minimized objective values."""
    started = time.perf_counter()
    rewards, u9 = objective.rewards(designs)
    batch_wall = time.perf_counter() - started
    rewards = as_numpy(rewards)
    log.add_batch(phase, np.atleast_2d(as_numpy(designs)), rewards, as_numpy(u9), batch_wall)
    return -rewards


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #
class _RestartBudgetExhausted(Exception):
    pass


def _run_ga_or_de(method: str, cfg: dict[str, Any], objective: SurrogateObjective,
                  log: SearchLog, deadline: float, seed: int) -> dict[str, Any]:
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.algorithms.soo.nonconvex.ga import GA, comp_by_cv_and_fitness
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.selection.tournament import TournamentSelection
    from pymoo.termination.max_gen import MaximumGenerationTermination

    class _Problem(Problem):
        def __init__(self):
            super().__init__(n_var=N_VARS, n_obj=1, xl=-1.0, xu=1.0)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = _evaluate(objective, log, method, np.asarray(X, dtype=np.float64))

    if method == "GA":
        ga = cfg["GA"]
        algorithm = GA(
            pop_size=int(ga["population"]),
            sampling=FloatRandomSampling(),
            selection=TournamentSelection(func_comp=comp_by_cv_and_fitness, pressure=2),
            crossover=SBX(prob=float(ga["crossover_prob"]), eta=float(ga["crossover_eta"])),
            mutation=PM(prob=float(ga["mutation_individual_prob"]),
                        prob_var=float(ga["mutation_variable_prob"]),
                        eta=float(ga["mutation_eta"])),
            eliminate_duplicates=bool(ga["eliminate_duplicates"]),
        )
    else:
        de = cfg["DE"]
        if de.get("dither") not in (None, "none", ""):
            raise ValueError(f"protocol DE.dither must be null, got {de.get('dither')!r}")
        if bool(de.get("jitter", False)):
            raise ValueError("protocol DE.jitter must be false")
        algorithm = DE(
            pop_size=int(de["population"]),
            sampling=FloatRandomSampling(),
            variant=str(de["variant"]),
            F=float(de["F"]),
            CR=float(de["CR"]),
            dither=None,
            jitter=False,
        )

    operators = _readback_operators(method, algorithm, cfg)
    problem = _Problem()
    algorithm.setup(problem, termination=MaximumGenerationTermination(10 ** 9),
                    seed=seed, verbose=False, save_history=False)
    generations = 0
    library_calls = 0
    while algorithm.has_next():
        if time.perf_counter() >= deadline:
            break
        algorithm.next()
        library_calls += 1
        generations += 1
    return {"generations": generations, "library_calls": library_calls,
            "population": int(algorithm.pop_size), "library_operators": operators}


def _value(obj: Any) -> Any:
    """pymoo wraps operator parameters in Real/Choice objects."""
    return getattr(obj, "value", obj)


def _readback_operators(method: str, algorithm: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    """Read the parameters back out of the constructed pymoo algorithm.

    Passing a value to a constructor is not evidence that the library used it.
    ``PM(prob=..)`` and ``PM(prob_var=..)`` in particular are distinct knobs and
    the protocol fixes both, so both are verified here and the run is refused on
    any mismatch.
    """
    if method == "GA":
        ga = cfg["GA"]
        mutation = algorithm.mating.mutation
        crossover = algorithm.mating.crossover
        got = {
            "population": int(algorithm.pop_size),
            "mutation_individual_prob": float(_value(mutation.prob)),
            "mutation_variable_prob": float(_value(mutation.prob_var)),
            "mutation_eta": float(_value(mutation.eta)),
            "crossover_prob": float(_value(crossover.prob)),
            "crossover_eta": float(_value(crossover.eta)),
            "selection_pressure": int(algorithm.mating.selection.pressure),
            "eliminate_duplicates": bool(algorithm.eliminate_duplicates is not None),
        }
        want = {
            "population": int(ga["population"]),
            "mutation_individual_prob": float(ga["mutation_individual_prob"]),
            "mutation_variable_prob": float(ga["mutation_variable_prob"]),
            "mutation_eta": float(ga["mutation_eta"]),
            "crossover_prob": float(ga["crossover_prob"]),
            "crossover_eta": float(ga["crossover_eta"]),
            "selection_pressure": 2,
            "eliminate_duplicates": bool(ga["eliminate_duplicates"]),
        }
    else:
        de = cfg["DE"]
        variant = algorithm.mating
        selection = str(_value(variant.selection))
        crossover = str(_value(variant.crossover))
        n_diffs = int(_value(variant.n_diffs))
        got = {
            "population": int(algorithm.pop_size),
            "F": float(_value(variant.F)),
            "CR": float(_value(variant.CR)),
            "jitter": bool(_value(variant.jitter)),
            "variant": f"DE/{selection}/{n_diffs}/{crossover}",
            "parameter_control": type(variant.control).__name__,
        }
        want = {
            "population": int(de["population"]),
            "F": float(de["F"]),
            "CR": float(de["CR"]),
            "jitter": False,
            "variant": str(de["variant"]),
            "parameter_control": "NoParameterControl",
        }
    mismatch = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if mismatch:
        raise ValueError(f"{method} operator read-back differs from protocol: {mismatch}")
    return {"observed": got, "protocol": want, "match": True}


def _run_lbfgsb(cfg: dict[str, Any], objective: SurrogateObjective, log: SearchLog,
                deadline: float, rng: np.random.Generator) -> dict[str, Any]:
    from scipy.optimize import minimize

    settings = cfg["LBFGSB"]
    per_restart = int(settings["per_restart_new_evaluations"])
    bounds = [(-1.0, 1.0)] * N_VARS
    restarts = 0
    library_calls = 0
    cache_hits = 0
    new_evaluations: list[int] = []

    while time.perf_counter() < deadline:
        cache: dict[bytes, tuple[float, np.ndarray]] = {}
        used = 0

        def fun(x: np.ndarray):
            nonlocal used, cache_hits
            vector = np.asarray(x, dtype=np.float64)
            key = vector.tobytes()
            hit = cache.get(key)
            if hit is not None:
                cache_hits += 1
                return hit[0], hit[1].copy()
            if used >= per_restart:
                raise _RestartBudgetExhausted
            batch_started = time.perf_counter()
            value, grad, u9 = objective.objective_and_grad(vector)
            batch_wall = time.perf_counter() - batch_started
            log.add_batch("LBFGSB", vector.reshape(1, N_VARS), np.array([-value]),
                          u9.reshape(1, 9), batch_wall)
            used += 1
            cache[key] = (value, grad)
            return value, grad.copy()

        x0 = (np.zeros(N_VARS, dtype=np.float64) if restarts == 0
              else rng.uniform(-1.0, 1.0, size=N_VARS))
        try:
            minimize(fun, x0, method="L-BFGS-B", jac=True, bounds=bounds,
                     options={"ftol": float(settings["ftol"]),
                              "gtol": float(settings["gtol"]),
                              "maxiter": int(settings["maxiter"]),
                              "maxfun": int(settings["maxfun"]),
                              "maxls": int(settings["maxls"])})
        except _RestartBudgetExhausted:
            pass
        library_calls += 1
        new_evaluations.append(used)
        restarts += 1

    return {"restarts": restarts, "library_calls": library_calls,
            "cache_hits": cache_hits, "new_evaluations_per_restart": new_evaluations,
            "library_operators": {"observed": {
                "per_restart_new_evaluations": per_restart,
                "ftol": float(settings["ftol"]), "gtol": float(settings["gtol"]),
                "maxiter": int(settings["maxiter"]), "maxfun": int(settings["maxfun"]),
                "maxls": int(settings["maxls"]), "jac": "autograd",
                "first_restart": "zero vector",
                "later_restarts": "uniform [-1,1] from the direct_search stream",
                "cache": "exact x bytes, value+gradient, restart-local",
            }, "protocol": dict(settings), "match": True}}


def _run_bo(cfg: dict[str, Any], objective: SurrogateObjective, log: SearchLog,
            deadline: float, seed: int) -> dict[str, Any]:
    import numpy as _np
    from skopt import Optimizer
    from skopt.learning import GaussianProcessRegressor
    from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern

    bo = cfg["BO"]
    kernel = (ConstantKernel(1.0, (0.01, 1000.0))
              * Matern(length_scale=_np.ones(N_VARS),
                       length_scale_bounds=(0.01, 100.0),
                       nu=2.5))
    estimator = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=bool(bo["normalize_y"]),
        noise="gaussian",
        n_restarts_optimizer=int(bo["gp_n_restarts_optimizer"]),
        random_state=seed,
    )
    optimizer = Optimizer(
        dimensions=[(-1.0, 1.0)] * N_VARS,
        base_estimator=estimator,
        n_initial_points=int(bo["initial_points"]),
        initial_point_generator=str(bo["initial_generator"]),
        acq_func=str(bo["acq_func"]),
        acq_optimizer=str(bo["acq_optimizer"]),
        acq_func_kwargs={"xi": float(bo["xi"]), "kappa": float(bo["kappa"])},
        acq_optimizer_kwargs={"n_points": int(bo["acq_n_points"]),
                              "n_restarts_optimizer": int(bo["acq_n_restarts_optimizer"]),
                              "n_jobs": int(bo["n_jobs"])},
        random_state=seed,
    )
    observed = {
        "initial_points": int(optimizer._n_initial_points),
        "acq_func": str(optimizer.acq_func),
        "acq_optimizer": str(optimizer.acq_optimizer),
        "xi": float(optimizer.acq_func_kwargs.get("xi")),
        "kappa": float(optimizer.acq_func_kwargs.get("kappa")),
        "acq_n_points": int(optimizer.acq_optimizer_kwargs.get("n_points")),
        "acq_n_restarts_optimizer": int(
            optimizer.acq_optimizer_kwargs.get("n_restarts_optimizer")),
        "n_jobs": int(optimizer.acq_optimizer_kwargs.get("n_jobs")),
        "normalize_y": bool(estimator.normalize_y),
        "noise": str(estimator.noise),
        "gp_n_restarts_optimizer": int(estimator.n_restarts_optimizer),
        "kernel": str(estimator.kernel),
        "n_dimensions": len(optimizer.space.dimensions),
    }
    want = {
        "initial_points": int(bo["initial_points"]),
        "acq_func": str(bo["acq_func"]),
        "acq_optimizer": str(bo["acq_optimizer"]),
        "xi": float(bo["xi"]),
        "kappa": float(bo["kappa"]),
        "acq_n_points": int(bo["acq_n_points"]),
        "acq_n_restarts_optimizer": int(bo["acq_n_restarts_optimizer"]),
        "n_jobs": int(bo["n_jobs"]),
        "normalize_y": bool(bo["normalize_y"]),
        "noise": "gaussian",
        "gp_n_restarts_optimizer": int(bo["gp_n_restarts_optimizer"]),
        "n_dimensions": N_VARS,
    }
    mismatch = {k: (observed[k], want[k]) for k in want if observed[k] != want[k]}
    if mismatch:
        raise ValueError(f"BO settings read-back differs from protocol: {mismatch}")

    asks = 0
    tells = 0
    while time.perf_counter() < deadline:
        x = optimizer.ask()
        asks += 1
        value = float(_evaluate(objective, log, "BO", _np.asarray(x, dtype=_np.float64))[0])
        optimizer.tell(list(x), value)
        tells += 1
    return {"library_calls": asks + tells, "asks": asks, "tells": tells,
            "n_initial_points": int(bo["initial_points"]),
            "library_operators": {"observed": observed, "protocol": want, "match": True}}


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def run_direct(case: dict[str, Any], attempt_dir: Path, device: Any,
               exclusive: bool | None = None,
               adam_backend: str | None = None,
               objective: SurrogateObjective | None = None,
               surrogate_source: str | None = None,
               protocol: dict[str, Any] | None = None,
               time_cap: dict[str, Any] | None = None,
               **_ignored: Any) -> dict[str, Any]:
    """Run one direct-optimization case and write its attempt artifacts.

    Trainer contract (``cmame_rt.run_case``)::

        run_direct(case, attempt_dir, device, exclusive=False, adam_backend=None)

    ``exclusive`` marks a measurement taken with the host to itself and is
    written to ``timing.json`` for the workstation timing-coverage report. When
    ``run_case`` does not pass it as a keyword it still reaches us on the case
    dict as ``_exclusive``, so both routes are honoured. ``adam_backend`` is
    recorded for provenance only; direct optimization runs no Adam step.

    ``objective`` is injected only by unit tests; a production run leaves it
    ``None`` so the paired transferred surrogate is loaded here and the clock
    starts immediately afterwards, as ``protocol.direct_optimization
    .timing_start`` requires.
    """
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol or _load_protocol()
    cfg = protocol["direct_optimization"]
    method = str(case["method"])
    if method not in METHODS:
        raise ValueError(f"unknown direct method {method!r}; expected {METHODS}")
    if str(cfg.get("domain")) != str(case.get("domain", cfg.get("domain"))):
        raise ValueError(
            f"case domain {case.get('domain')!r} != protocol direct domain {cfg.get('domain')!r}"
        )
    cap = time_cap or load_time_cap()
    seed = int(case["seed"])
    stream_seed = subseed(seed, DIRECT_SEARCH_STREAM)
    rng = np.random.default_rng(stream_seed)

    if exclusive is None:
        exclusive = case.get("_exclusive", False)
    exclusive = bool(exclusive)
    if adam_backend is None:
        adam_backend = case.get("_adam_backend")

    started_at = _utc_now()
    time_cap_s = float(cap["time_cap_seconds"])

    # ---- surrogate loading is measured but is outside the capped window
    _sync(device)
    t_load_start = time.perf_counter()
    surrogate_loaded_here = objective is None
    if objective is None:
        objective, surrogate_source = build_objective(case, device)
    _sync(device)
    t_load_s = time.perf_counter() - t_load_start

    # ---- clock starts here: surrogate is loaded, nothing has been initialized
    t0 = time.perf_counter()
    deadline = t0 + time_cap_s
    objective_name = str(case.get("objective", "arithmetic"))
    log = SearchLog(attempt_dir / "search_log.csv", t0, deadline,
                    objective.goal, objective_name)
    try:
        if method in ("GA", "DE"):
            method_counts = _run_ga_or_de(method, cfg, objective, log, deadline, stream_seed)
        elif method == "LBFGSB":
            method_counts = _run_lbfgsb(cfg, objective, log, deadline, rng)
        else:
            method_counts = _run_bo(cfg, objective, log, deadline, stream_seed)
    finally:
        _sync(device)
        t_total_s = time.perf_counter() - t0
        log.close()
    overrun_s = max(0.0, t_total_s - time_cap_s)
    t_search_s = min(t_total_s, time_cap_s)

    incumbent = log.incumbent()
    goal = objective.goal
    bounds = protocol["domains"][cfg["domain"]]["thickness_bounds_mm"]
    if incumbent is not None:
        actions = as_numpy(incumbent["actions30"])
        t_min, t_max = float(bounds[0]), float(bounds[1])
        incumbent["physical_thickness30"] = (
            t_min + (actions + 1.0) * (t_max - t_min) / 2.0
        ).tolist()
        incumbent["goal"] = goal
        incumbent["domain"] = cfg["domain"]
        incumbent["thickness_bounds_mm"] = [t_min, t_max]
    finished_at = _utc_now()

    library_operators = method_counts.pop("library_operators", None)
    n_within_cap = sum(1 for r in log.rows if r["within_cap"])
    host = socket.gethostname()
    counts = {
        # fixed key names (coordinator, 2026-09-13 15:45)
        "surrogate_value_calls": int(objective.counts["surrogate_value_calls"]),
        "surrogate_grad_calls": int(objective.counts["surrogate_grad_calls"]),
        "library_calls": int(method_counts.get("library_calls", 0)),
        "cache_hits": int(method_counts.get("cache_hits", 0)),
        "candidates_total": int(log.index),
        "candidates_within_cap": int(n_within_cap),
        "batch_eval_time_s": float(log.batch_eval_time_s),
        "overrun_candidates": int(log.n_overrun),
        # diagnostics kept alongside the fixed keys
        "surrogate_value_candidates": int(objective.counts["surrogate_value_candidates"]),
        "surrogate_grad_candidates": int(objective.counts["surrogate_grad_candidates"]),
        "batches": int(log.batch_index),
        "method_detail": {k: v for k, v in method_counts.items()
                          if k not in ("library_calls", "cache_hits")},
    }
    timing = {
        # fixed key names (coordinator, 2026-09-13 15:45)
        "t_load_s": float(t_load_s),
        "t_total_s": float(t_total_s),
        "t_search_s": float(t_search_s),
        "time_cap_s": float(time_cap_s),
        "overrun_s": float(overrun_s),
        "n_candidates_within_cap": int(n_within_cap),
        "exclusive": bool(exclusive),
        "host": host,
        "started_at": started_at,
        "finished_at": finished_at,
        # provenance and diagnostics
        "time_cap_seconds": float(time_cap_s),
        "time_cap_file": cap["file"],
        "time_cap_sha256": cap["sha256"],
        "t_last_within_cap_candidate_s": float(log.last_within_cap_elapsed_s),
        "surrogate_loaded_in_run_direct": bool(surrogate_loaded_here),
        "adam_backend": adam_backend,
        "device": str(device),
        "timing_start": cfg["timing_start"],
        "timing_scope": cfg["timing_scope"],
        "t_total_definition": "perf_counter from timing_start to the end of the search, "
                              "CUDA-synchronized at both ends; equals t_search_s + overrun_s",
    }
    missing_timing = _missing_or_null(timing, REQUIRED_TIMING_KEYS)
    missing_counts = _missing_or_null(counts, REQUIRED_COUNTS_KEYS)
    resolved = {
        "case": case,
        "method": method,
        "method_settings": cfg[method],
        "library_operators_readback": library_operators,
        "domain": cfg["domain"],
        "bounds": cfg["bounds"],
        "variables": cfg["variables"],
        "terminal_objective": (cfg["terminal_objective"]
                               if objective_name == "arithmetic" else
                               "negative bilateral reward"),
        "reporting_objective": "canonical arithmetic reward",
        "C1": objective.C1,
        "C2": objective.C2,
        "goal": goal,
        "seed": seed,
        "rng_stream": "direct_search",
        "rng_stream_id": DIRECT_SEARCH_STREAM,
        "rng_subseed": stream_seed,
        "surrogate_source": surrogate_source,
        "protocol_file_sha256": _sha256_file(_root() / "spec" / "protocol.json"),
        "protocol_canonical_json_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "time_cap": cap,
        "exclusive": bool(exclusive),
        "adam_backend": adam_backend,
        "host": host,
        "device": str(device),
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    _write_json(attempt_dir / "resolved_config.json", resolved)
    _write_json(attempt_dir / "incumbent.json", incumbent or {"incumbent": None})
    _write_json(attempt_dir / "counts.json", counts)
    _write_json(attempt_dir / "timing.json", timing)

    artifacts = {name: _sha256_file(attempt_dir / name)
                 for name in ("search_log.csv", "resolved_config.json", "incumbent.json",
                              "counts.json", "timing.json")}
    complete = incumbent is not None and not missing_timing and not missing_counts
    result = {
        "complete": bool(complete),
        "case_id": case["case_id"],
        "attempt": int(case.get("attempt", 0)),
        "kind": "optimizer",
        "method": method,
        "objective": objective_name,
        "nonfinite": bool(incumbent is None
                          or not np.isfinite(incumbent["canonical_reward"])
                          or not np.isfinite(incumbent["training_objective"])),
        "candidates_within_cap": counts["candidates_within_cap"],
        "candidates_overrun": counts["overrun_candidates"],
        "best_canonical_reward": (None if incumbent is None
                                   else incumbent["canonical_reward"]),
        "best_training_objective": (None if incumbent is None
                                    else incumbent["training_objective"]),
        "artifact_sha256": artifacts,
        "t_load_s": float(t_load_s),
        "t_total_s": float(t_total_s),
        "t_search_s": float(t_search_s),
        "time_cap_s": float(time_cap_s),
        "overrun_s": float(overrun_s),
        "exclusive": bool(exclusive),
        "host": host,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    if missing_timing or missing_counts:
        # A silent gap in the timing record is a failure, not a completed run.
        result["error_type"] = "IncompleteTimingRecord"
        result["error"] = (
            "required fields are missing or null: "
            f"timing.json {missing_timing}, counts.json {missing_counts}"
        )
        result["missing_timing_keys"] = missing_timing
        result["missing_counts_keys"] = missing_counts
    _write_json(attempt_dir / "result.json", result)
    return result


def _write_json(path: Path, payload: Any) -> None:
    tmp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "torch", "pymoo", "skopt", "sklearn"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "absent"
    return versions
