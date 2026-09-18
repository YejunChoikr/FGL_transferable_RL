"""Run the registered CMAME experiments without the multi-machine scheduler."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cmame_rt.protocol import PROTOCOL_HASH, case_by_id, load_cases


def required_dependencies(case):
    deps = case.get("dependencies", [])
    # The three policy runs freeze the published time allowance. It is supplied
    # in locks/DIRECT_TIME_CAP.json, so optimizers only need the surrogate here.
    return [d for d in deps if case["kind"] != "optimizer" or d.startswith("surrogate/")]


def plan(case_id):
    order, seen = [], set()
    def visit(cid):
        if cid in seen:
            return
        seen.add(cid)
        for dep in required_dependencies(case_by_id(cid)):
            visit(dep)
        order.append(cid)
    visit(case_id)
    return order


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_digest():
    files = sorted((ROOT / "cmame_rt").glob("*.py")) + [Path(__file__)]
    return hashlib.sha256("".join(sha256(p) for p in files).encode()).hexdigest()


def validate_completed(path, case_id, smoke):
    marker = path / "completed.json"
    if not marker.exists():
        return False
    d = json.loads(marker.read_text(encoding="utf-8"))
    if (d["case_id"] != case_id or d["protocol_hash"] != PROTOCOL_HASH
            or d["smoke"] != smoke or d["code_sha256"] != code_digest()):
        raise RuntimeError(f"configuration changed for {path}; preserve it and use a separate checkout")
    for name, expected in d["artifact_sha256"].items():
        if not (path/name).is_file() or sha256(path/name) != expected:
            raise RuntimeError(f"missing or modified artifact: {path/name}")
    return True


def prepare_initialization(case):
    import numpy as np
    import torch
    if case["kind"] == "supervised":
        from cmame_rt import init_bank as bank
        arch, seed = case["architecture"], int(case["seed"])
        path = bank.init_path(arch, seed)
        if not path.exists():
            bank.save_init(bank.build_surrogate_init(seed, arch), arch, seed)
    elif case["kind"] == "policy":
        from cmame_rt import policy_init as bank
        algo, seed = case["algorithm"], int(case["seed"])
        nets = ("actor", "q1", "q2") if algo == "SAC" else ("actor", "critic")
        if not all(bank.init_path(algo, n, seed).exists() for n in nets):
            for net, sd in bank.build_policy_init(seed, algo).items():
                path = bank.init_path(algo, net, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(sd, path)
        p = bank.prefill_path(seed)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, bank.build_prefill(seed))


def run_one(case_id, device, backend, with_dependencies=False, smoke=False):
    import torch
    from cmame_rt.data import verify_data_lock
    case = case_by_id(case_id)
    output_root = ROOT / ("pilot/smoke" if smoke else "accepted")
    out = output_root / case_id
    if validate_completed(out, case_id, smoke):
        print(f"Already complete: {case_id}", flush=True)
        return out
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"incomplete output exists at {out}; inspect it before retrying")
    deps = required_dependencies(case)
    for dep in deps:
        if with_dependencies:
            run_one(dep, device, backend, True, smoke)
        elif not validate_completed(output_root / dep, dep, smoke):
            raise RuntimeError(f"missing dependency {dep}; run it first or use --with-dependencies")
    if not verify_data_lock()["ok"]:
        raise RuntimeError("the frozen input data do not match their manifest")
    prepare_initialization(case)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'SMOKE' if smoke else 'RUN'} {case_id}", flush=True)
    if case["kind"] == "supervised":
        from cmame_rt.surrogate_train import run_supervised
        source = output_root / deps[0] / "best.pt" if deps else None
        result = run_supervised(case, out, device=device, adam_backend=backend,
                                source_best_path=source, epochs=1 if smoke else None)
        from cmame_rt.checkpoint_evaluation import evaluate_checkpoint
        for which in ("best", "last"):
            evaluate_checkpoint(out, which, device=device)
    elif case["kind"] == "policy":
        from cmame_rt.policy_train import run_policy
        from cmame_rt.reporting import summarize_run
        sdir = output_root / next(d for d in deps if d.startswith("surrogate/"))
        source = next((output_root / d for d in deps if d.startswith("policy/")), None)
        paths = ({"actor": source / "best_actor.pt", "critics": source / "best_critics.pt"}
                 if source is not None else None)
        result = run_policy(case, out, device=device, adam_backend=backend,
                            surrogate_path=sdir / "best.pt", scaler_path=sdir / "scaler.json",
                            source_paths=paths, exclusive=False, resume_enabled=False,
                            episodes=25 if smoke else None,
                            eval_episodes=[0, 25] if smoke else None)
        summarize_run(out, case["goals"])
    else:
        from cmame_rt.direct import run_direct, load_time_cap, SurrogateObjective
        from cmame_rt.surrogate_train import load_surrogate_for_env
        from cmame_rt import encoding
        from cmame_rt.reward import reward_torch
        sdir = output_root / next(d for d in deps if d.startswith("surrogate/"))
        model, scaler = load_surrogate_for_env(sdir, device)
        objective = SurrogateObjective(model=model, y_mean=scaler["mean"],
            y_scale=scaler["scale"], goal=case["goals"][0], device=device,
            C1=case["C1"], C2=case["C2"],
            encode_batch_torch=encoding.encode_batch_torch,
            to_cnn_input=encoding.to_cnn_input, reward_torch=reward_torch)
        cap = load_time_cap()
        if smoke:
            cap = dict(cap, time_cap_seconds=1.0, time_cap_s=1.0,
                       smoke_override=True)
        result = run_direct(case, out, device=device, exclusive=False,
                            objective=objective, surrogate_source=str(sdir), time_cap=cap)
    if not result.get("complete", False) or result.get("nonfinite", False):
        raise RuntimeError(f"run did not complete successfully: {out}")
    payload = {"case_id": case_id, "protocol_hash": PROTOCOL_HASH,
               "code_sha256": code_digest(), "smoke": smoke,
               "device": str(device), "adam_backend": backend,
               "artifact_sha256": {p.name: sha256(p) for p in out.iterdir() if p.is_file()}}
    (out / "completed.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Completed: {out}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("list", help="list registered experiments")
    ls.add_argument("--kind", choices=["supervised", "policy", "optimizer"])
    ls.add_argument("--domain", choices=["source", "upper", "lower", "ar08"])
    ls.add_argument("--seed", type=int)
    ls.add_argument("--contains", default="")
    for command in ["run", "smoke", "plan"]:
        q = sub.add_parser(command)
        q.add_argument("--case", required=True)
        if command != "plan":
            q.add_argument("--device", default="auto")
            q.add_argument("--adam-backend", choices=["auto", "reference", "foreach", "fused"], default="auto")
            q.add_argument("--with-dependencies", action="store_true")
    sub.add_parser("verify", help="verify the frozen data and experiment definitions")
    args = ap.parse_args(argv)
    if args.command == "list":
        for c in load_cases():
            if all(getattr(args, k) is None or c[k] == getattr(args, k)
                   for k in ["kind", "domain", "seed"]) and args.contains in c["case_id"]:
                print(c["case_id"])
        return
    if args.command == "plan":
        print("\n".join(plan(args.case)))
        return
    if args.command == "verify":
        from cmame_rt.data import verify_data_lock
        from cmame_rt.protocol import hash_report
        result = {"protocol": hash_report(), "data": verify_data_lock(), "case_count": len(load_cases())}
        print(json.dumps(result, indent=2))
        if not result["data"]["ok"] or not result["protocol"]["protocol_hash_matches_spec"]:
            raise SystemExit(1)
        return
    import torch
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    backend = ("fused" if str(device).startswith("cuda") else "reference") if args.adam_backend == "auto" else args.adam_backend
    run_one(args.case, device, backend, args.with_dependencies or args.command == "smoke", args.command == "smoke")


if __name__ == "__main__":
    main()
