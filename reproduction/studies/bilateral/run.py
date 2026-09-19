"""Run the S10 bilateral cases through the common CMAME runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
STUDY_ROOT = Path(__file__).resolve().parent
REPRO_ROOT = STUDY_ROOT.parents[1]
sys.path.insert(0, str(REPRO_ROOT))

import run as common_run
from cmame_rt.protocol import PROTOCOL_HASH

BO_TIME_CAP_S = 382.5083336830139
GOALS = (5, 6, 7)
SEEDS = tuple(range(5))


def cases() -> list[dict]:
    """Return 30 source policies, 60 target policies and 15 BO cases."""
    rows = []
    for algorithm in ("SAC", "DDPG"):
        for goal in GOALS:
            for seed in SEEDS:
                task = f"FGL{goal}"
                source_id = f"source/{task}/{algorithm}-SOURCE/s{seed}"
                common = {
                    "kind": "policy", "algorithm": algorithm, "seed": seed,
                    "goals": [goal], "C1": 0.5, "C2": 1.0,
                    "traversal": "rowwise_raster", "shared": False,
                    "objective": "bilateral", "group": "S10_bilateral",
                }
                rows.append({**common, "case_id": source_id,
                             "domain": "source", "arm": "P0",
                             "dependencies": [
                                 f"surrogate/cnn/source/N30000/scratch/s{seed}"]})
                rows.append({**common,
                             "case_id": f"upper/{task}/{algorithm}-RL/s{seed}",
                             "domain": "upper", "arm": "ST_P0",
                             "dependencies": [
                                 f"surrogate/cnn/upper/N6000/transfer/s{seed}"]})
                rows.append({**common,
                             "case_id": f"upper/{task}/{algorithm}-TRL/s{seed}",
                             "domain": "upper", "arm": "ST_PAC",
                             "dependencies": [
                                 f"surrogate/cnn/upper/N6000/transfer/s{seed}",
                                 source_id]})
    for goal in GOALS:
        for seed in SEEDS:
            rows.append({
                "case_id": f"upper/FGL{goal}/BO/s{seed}",
                "kind": "optimizer", "method": "BO", "domain": "upper",
                "goal": goal, "goals": [goal], "seed": seed,
                "C1": 0.5, "C2": 1.0, "objective": "bilateral",
                "dependencies": [
                    f"surrogate/cnn/upper/N6000/transfer/s{seed}"],
            })
    return rows


def case_map() -> dict[str, dict]:
    return {case["case_id"]: case for case in cases()}


def plan(case_id: str) -> list[str]:
    table = case_map()
    order, seen = [], set()

    def add(cid: str) -> None:
        if cid in seen:
            return
        if cid in table:
            for dependency in table[cid]["dependencies"]:
                add(dependency)
        else:
            for dependency in common_run.plan(cid):
                if dependency not in seen:
                    seen.add(dependency)
                    order.append(dependency)
            return
        seen.add(cid)
        order.append(cid)

    if case_id not in table:
        raise KeyError(f"unknown S10 case: {case_id}")
    add(case_id)
    return order


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _code_digest() -> str:
    files = sorted((REPRO_ROOT / "cmame_rt").glob("*.py")) + [Path(__file__)]
    return hashlib.sha256("".join(_sha256(path) for path in files).encode()).hexdigest()


def _output_root(smoke: bool) -> Path:
    return (REPRO_ROOT / "pilot" / "smoke" / "s10" if smoke
            else STUDY_ROOT / "generated")


def _main_output_root(smoke: bool) -> Path:
    return REPRO_ROOT / ("pilot/smoke" if smoke else "accepted")


def _completed(path: Path, case_id: str, smoke: bool) -> bool:
    marker = path / "completed.json"
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    expected = {"case_id": case_id, "smoke": smoke,
                "protocol_hash": PROTOCOL_HASH, "code_sha256": _code_digest()}
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"configuration changed for {path}; use a new checkout")
    for name, digest in record["artifact_sha256"].items():
        artifact = path / name
        if not artifact.is_file() or _sha256(artifact) != digest:
            raise RuntimeError(f"missing or modified artifact: {artifact}")
    return True


def _ensure_main_dependency(case_id: str, device: str, backend: str,
                            with_dependencies: bool, smoke: bool) -> Path:
    out = _main_output_root(smoke) / case_id
    marker = out / "completed.json"
    if marker.is_file():
        record = json.loads(marker.read_text(encoding="utf-8"))
        required = ("best.pt", "scaler.json")
        reusable = (record.get("case_id") == case_id
                    and record.get("protocol_hash") == PROTOCOL_HASH
                    and record.get("smoke") == smoke
                    and all(name in record.get("artifact_sha256", {})
                            for name in required)
                    and all((out / name).is_file()
                            and _sha256(out / name)
                            == record["artifact_sha256"][name]
                            for name in required))
        if reusable:
            return out
    if with_dependencies:
        return common_run.run_one(case_id, device, backend, True, smoke)
    if not common_run.validate_completed(out, case_id, smoke):
        raise RuntimeError(f"missing dependency {case_id}; use --with-dependencies")
    return out


def execute(case_id: str, device: str, backend: str,
            with_dependencies: bool = False, smoke: bool = False,
            exclusive: bool = False) -> Path:
    """Execute one case while resolving all models through the main registry."""
    table = case_map()
    case = table[case_id]
    out = _output_root(smoke) / Path(case_id)
    if _completed(out, case_id, smoke):
        print(f"Already complete: {case_id}", flush=True)
        return out
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"incomplete output exists at {out}; inspect it before retrying")

    dependency_paths = {}
    for dependency in case["dependencies"]:
        if dependency in table:
            if with_dependencies:
                dependency_paths[dependency] = execute(
                    dependency, device, backend, True, smoke, exclusive)
            else:
                dep_out = _output_root(smoke) / Path(dependency)
                if not _completed(dep_out, dependency, smoke):
                    raise RuntimeError(
                        f"missing dependency {dependency}; use --with-dependencies")
                dependency_paths[dependency] = dep_out
        else:
            dependency_paths[dependency] = _ensure_main_dependency(
                dependency, device, backend, with_dependencies, smoke)

    common_run.prepare_initialization(case)
    out.mkdir(parents=True, exist_ok=True)
    if case["kind"] == "policy":
        from cmame_rt.policy_train import run_policy
        from cmame_rt.reporting import summarize_run

        surrogate_id = next(d for d in case["dependencies"]
                            if d.startswith("surrogate/"))
        surrogate_dir = dependency_paths[surrogate_id]
        source_id = next((d for d in case["dependencies"] if d in table), None)
        source_paths = None
        if source_id is not None:
            source_dir = dependency_paths[source_id]
            source_paths = {"actor": source_dir / "best_actor.pt",
                            "critics": source_dir / "best_critics.pt"}
        result = run_policy(
            case, out, device=device, adam_backend=backend,
            surrogate_path=surrogate_dir / "best.pt",
            scaler_path=surrogate_dir / "scaler.json",
            source_paths=source_paths, exclusive=False, resume_enabled=False,
            episodes=25 if smoke else None,
            eval_episodes=[0, 25] if smoke else None)
        summarize_run(out, case["goals"])
    else:
        import torch
        from cmame_rt import encoding
        from cmame_rt.direct import (SurrogateObjective, load_time_cap,
                                     run_direct)
        from cmame_rt.reward import bilateral_reward_torch
        from cmame_rt.surrogate_train import load_surrogate_for_env

        surrogate_id = case["dependencies"][0]
        surrogate_dir = dependency_paths[surrogate_id]
        model, scaler = load_surrogate_for_env(surrogate_dir, device)
        objective = SurrogateObjective(
            model=model, y_mean=scaler["mean"], y_scale=scaler["scale"],
            goal=case["goal"], C1=case["C1"], C2=case["C2"], device=device,
            encode_batch_torch=encoding.encode_batch_torch,
            to_cnn_input=encoding.to_cnn_input,
            reward_torch=bilateral_reward_torch)
        cap = load_time_cap()
        if smoke:
            cap = dict(cap, time_cap_seconds=1.0, time_cap_s=1.0,
                       smoke_override=True)
        else:
            if not exclusive:
                raise RuntimeError("full S10 BO requires --exclusive")
            gpu = torch.cuda.get_device_name(torch.device(device))
            if "4080 SUPER" not in gpu:
                raise RuntimeError("full S10 BO requires an RTX 4080 SUPER")
            if float(cap["time_cap_seconds"]) != BO_TIME_CAP_S:
                raise RuntimeError("DIRECT_TIME_CAP.json is not the S10 common cap")
        result = run_direct(
            case, out, device=device, exclusive=exclusive and not smoke,
            objective=objective, surrogate_source=str(surrogate_dir), time_cap=cap)

    if not result.get("complete", False) or result.get("nonfinite", False):
        raise RuntimeError(f"run did not complete successfully: {out}")
    artifacts = {path.name: _sha256(path) for path in out.iterdir()
                 if path.is_file()}
    marker = {"case_id": case_id, "protocol_hash": PROTOCOL_HASH,
              "code_sha256": _code_digest(), "smoke": smoke,
              "device": str(device), "adam_backend": backend,
              "objective": "bilateral", "artifact_sha256": artifacts}
    (out / "completed.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(f"Completed: {out}", flush=True)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list the 105 S10 cases")
    listing.add_argument("--contains", default="")
    for command in ("run", "smoke", "plan"):
        item = sub.add_parser(command)
        item.add_argument("--case", required=True)
        if command != "plan":
            item.add_argument("--device", default="auto")
            item.add_argument("--adam-backend",
                              choices=["auto", "reference", "foreach", "fused"],
                              default="auto")
            item.add_argument("--with-dependencies", action="store_true")
            item.add_argument("--exclusive", action="store_true",
                              help="assert that the BO workstation is idle")
    args = parser.parse_args(argv)
    if args.command == "list":
        for case in cases():
            if args.contains in case["case_id"]:
                print(case["case_id"])
        return
    if args.command == "plan":
        print("\n".join(plan(args.case)))
        return

    import torch
    device = (("cuda:0" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    backend = (("fused" if str(device).startswith("cuda") else "reference")
               if args.adam_backend == "auto" else args.adam_backend)
    smoke = args.command == "smoke"
    execute(args.case, device, backend,
            with_dependencies=args.with_dependencies or smoke,
            smoke=smoke, exclusive=args.exclusive)


if __name__ == "__main__":
    main()
