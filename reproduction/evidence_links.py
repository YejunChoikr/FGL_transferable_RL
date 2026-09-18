"""Validate shared policy records and saved solver output without running experiments."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np


def read_json(archive, name):
    return json.loads(archive.read(name))


def coefficient_cases(evidence, registry):
    """Locate every sequential coefficient condition also stored in policies.zip."""
    sweeps = json.loads(Path(registry).read_text(encoding="utf-8"))
    unique = {r["case_id"]: r for stage in ["C1", "C2"] for r in sweeps[stage]}
    rows = []
    with zipfile.ZipFile(Path(evidence)/"policies.zip") as policies:
        names = set(policies.namelist())
        for cid, r in sorted(unique.items()):
            if cid + "/selected_designs.json" not in names:
                continue
            stage = "stage1_C1" if r["C2"] == 0 else "stage2_C2"
            prefix = (f"runs/{stage}/FGL{r['goal']}/"
                      f"C1_{r['C1']:g}__C2_{r['C2']:g}/seed_{r['seed']}")
            rows.append({**r, "prefix": prefix})
    return rows


def verify_coefficient_links(evidence, registry):
    """Require identical designs, objectives, episodes, curves and configurations."""
    expected = coefficient_cases(evidence, registry)
    with zipfile.ZipFile(Path(evidence)/"coefficients.zip") as coef, \
            zipfile.ZipFile(Path(evidence)/"policies.zip") as policies:
        links = read_json(coef, "POLICY_LINKS.json")
        assert len(links) == len(expected), "incomplete coefficient-policy links"
        by_case = {r["case_id"]: r for r in links}
        for row in expected:
            cid, prefix = row["case_id"], row["prefix"]
            link = by_case[cid]
            assert link["coefficient_prefix"] == prefix
            for member, digest in link["policy_member_sha256"].items():
                assert hashlib.sha256(policies.read(member)).hexdigest() == digest, member
            record = read_json(coef, prefix + "/best_det_design.json")
            selected = read_json(policies, cid + "/selected_designs.json")
            design, = selected["designs"]
            np.testing.assert_array_equal(record["action"], design["actions30"])
            np.testing.assert_array_equal(record["u"], design["u9_pred"])
            assert record["reward_env"] == design["training_objective"], cid
            assert record["episode"] == selected["selected_episode"], cid
            assert coef.read(prefix + "/resolved_config.json") == policies.read(cid + "/resolved_config.json"), cid
            log = list(csv.DictReader(io.StringIO(policies.read(cid + "/episode_log.csv").decode())))
            history = np.load(io.BytesIO(coef.read(prefix + "/reward_history.npy")), allow_pickle=False)
            np.testing.assert_array_equal(history, [float(r["training_reward"]) for r in log])
        summaries = list(csv.DictReader(io.StringIO(coef.read("results/run_summary.csv").decode("utf-8-sig"))))
        for row in summaries:
            prefix = row["artifact_path"].replace("\\", "/")
            d = read_json(coef, prefix + "/best_det_design.json")
            assert float(row["best_det_reward"]) == d["reward_env"], prefix
            assert int(row["best_det_episode"]) == d["episode"], prefix
            np.testing.assert_array_equal([float(row[f"action_{i:02d}"]) for i in range(1,31)], d["action"])
            np.testing.assert_array_equal([float(row[f"u{i}"]) for i in range(1,10)], d["u"])
        for name in ["results/C1_summary.csv", "results/C2_summary.csv"]:
            for row in csv.DictReader(io.StringIO(coef.read(name).decode("utf-8-sig"))):
                c1, c2 = float(row["C1"]), float(row["C2"])
                stage = "stage1_C1" if c2 == 0 else "stage2_C2"
                group = [read_json(coef, f"runs/{stage}/{row['task']}/C1_{c1:g}__C2_{c2:g}/seed_{s}/best_det_design.json") for s in range(5)]
                for key in ["kappa", "rho_A", "bilateral_raw", "bilateral_normalized", "global_margin", "reward_env", "own_objective_reward", "original_locked_reward_at_C1_0.5_C2_1.0"]:
                    a = [d[key] for d in group]
                    np.testing.assert_allclose(float(row[key+"_mean"]), np.mean(a), rtol=0, atol=1e-12)
                    np.testing.assert_allclose(float(row[key+"_std"]), np.std(a, ddof=1), rtol=0, atol=1e-12)
    return {"shared_cases": len(expected), "matching_cases": len(expected), "designs_curves_and_summaries_match": True}


def parse_solver_values(text):
    pairs = re.findall(r"^\s*(solve_converged|r_force|top_uy|xr[1-9])\s*=\s*([-+\d.EeDd]+)", text, re.M)
    result = {key: float(value.replace("D", "E").replace("d", "e")) for key, value in pairs}
    assert len(result) == 12, "missing solver quantities"
    assert result["solve_converged"] == 1, "solver did not converge"
    assert abs(result["top_uy"] + 14.999) <= 1e-8, "incorrect applied compression"
    return result


def verify_solver_record(archive, record):
    """Tie selected actions, thicknesses, nodal output and final probe values together."""
    for member, digest in record["files_sha256"].items():
        assert hashlib.sha256(archive.read(member)).hexdigest() == digest, member
    selected = read_json(archive, record["selected_design_member"])
    np.testing.assert_array_equal(record["actions30"], selected["actions"])
    thickness = [float(row[f"col{i}_mm"]) for row in csv.DictReader(io.StringIO(archive.read(record["thickness_member"]).decode("utf-8-sig"))) for i in range(1,4)]
    np.testing.assert_allclose(thickness, selected["thickness_mm"], rtol=0, atol=1e-12)
    values = parse_solver_values(archive.read(record["result_member"]).decode())
    profile = [values[f"xr{i}"] for i in range(1,10)]
    geometry = archive.read(record["geometry_member"]).decode()
    nodes = {}
    for line in geometry.splitlines():
        fields = line.split()
        if len(fields) == 6 and fields[0] == "N":
            _, _, x, y, ux, _ = fields
            x, y, ux = float(x), float(y), float(ux)
            if abs(x-30) < 1e-9 and y in range(10,100,10):
                nodes[int(y/10)] = ux
    np.testing.assert_allclose(profile, [nodes[i] for i in range(1,10)], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(profile, record["profile"])
    return profile


if __name__ == "__main__":
    import argparse
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=root/"evidence")
    args = parser.parse_args()
    index = json.loads((args.evidence/"INDEX.json").read_text(encoding="utf-8"))
    for name in ["coefficients.zip", "policies.zip", "bilateral.zip"]:
        assert hashlib.sha256((args.evidence/name).read_bytes()).hexdigest() == index["archives"][name]["sha256"], name
        with zipfile.ZipFile(args.evidence/name) as archive:
            for member, meta in read_json(archive, "MANIFEST.json").items():
                data = archive.read(member)
                assert hashlib.sha256(data).hexdigest() == meta["sha256"], member
                assert len(data) == meta["bytes"], member
    report = {"coefficient_policy_links": verify_coefficient_links(
        args.evidence, root/"reproduction/spec/coefficient_sweeps.json")}
    with zipfile.ZipFile(args.evidence/"bilateral.zip") as archive:
        records = read_json(archive, "solver_records.json")
        rows = read_json(archive, "final_upper_runs.json")
        raw = read_json(archive, "ddpg_c4_fea.json")
        supplied = {(r["task"], r["method"], r["seed"]) for r in records}
        for row in rows:
            if row.get("solver_record_member"):
                assert (row["task"], row["method"], int(row["seed"])) in supplied, "missing solver evidence"
        for record in records:
            profile = verify_solver_record(archive, record)
            row = next(r for r in rows if (r["task"], r["method"], int(r["seed"])) == (record["task"], record["method"], record["seed"]))
            fea = next(r for r in raw if (r["task"], int(r["seed"])) == (record["task"], record["seed"]))
            np.testing.assert_array_equal(profile, [float(row[f"fea_u{i}"]) for i in range(1,10)])
            np.testing.assert_array_equal(profile, [float(fea[f"u{i}"]) for i in range(1,10)])
            assert fea["status"] == "success"
            a = np.asarray(record["actions30"])
            rho = float(np.mean((a+1)/2)); i = int(record["task"][-1])-1
            reward = (profile[i]-.5*(profile[i-1]+profile[i+1]))/(.5+rho)
            margin = min(profile[i]-profile[i-1], profile[i]-profile[i+1])
            assert abs(float(row["fea_original_reward"])-reward) < 1e-12
            assert abs(float(row["fea_bilateral_margin"])-margin) < 1e-12
            assert abs(float(row["fea_bilateral_reward"])-margin/(.5+rho)) < 1e-12
        report["solver_evidence"] = {"verified_records":len(records), "design_nodal_profile_and_aggregate_input_match":True}
    print(json.dumps(report, indent=2))
