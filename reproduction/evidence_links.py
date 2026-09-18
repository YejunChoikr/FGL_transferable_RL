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
        missing = sorted(cid for cid in unique if cid + "/selected_designs.json" not in names)
        assert not missing, f"missing sequential coefficient policy cases: {missing}"
        for cid, r in sorted(unique.items()):
            stage = "stage1_C1" if r["C2"] == 0 else "stage2_C2"
            prefix = (f"runs/{stage}/FGL{r['goal']}/"
                      f"C1_{r['C1']:g}__C2_{r['C2']:g}/seed_{r['seed']}")
            rows.append({**r, "prefix": prefix})
    assert len(rows) == 135, "sequential registry must contain 135 unique coefficient cases"
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
        summaries = list(csv.DictReader(io.StringIO(coef.read("results/surrogate_run_summary.csv").decode("utf-8-sig"))))
        for row in summaries:
            prefix = row["artifact_path"].replace("\\", "/")
            d = read_json(coef, prefix + "/best_det_design.json")
            assert float(row["best_det_reward"]) == d["reward_env"], prefix
            assert int(row["best_det_episode"]) == d["episode"], prefix
            np.testing.assert_array_equal([float(row[f"action_{i:02d}"]) for i in range(1,31)], d["action"])
            np.testing.assert_array_equal([float(row[f"u{i}"]) for i in range(1,10)], d["u"])
        for name in ["results/surrogate_C1_summary.csv", "results/surrogate_C2_summary.csv"]:
            for row in csv.DictReader(io.StringIO(coef.read(name).decode("utf-8-sig"))):
                c1, c2 = float(row["C1"]), float(row["C2"])
                stage = "stage1_C1" if c2 == 0 else "stage2_C2"
                group = [read_json(coef, f"runs/{stage}/{row['task']}/C1_{c1:g}__C2_{c2:g}/seed_{s}/best_det_design.json") for s in range(5)]
                for key in ["kappa", "rho_A", "bilateral_raw", "bilateral_normalized", "global_margin", "reward_env", "own_objective_reward", "original_locked_reward_at_C1_0.5_C2_1.0"]:
                    a = [d[key] for d in group]
                    np.testing.assert_allclose(float(row[key+"_mean"]), np.mean(a), rtol=0, atol=1e-12)
                    np.testing.assert_allclose(float(row[key+"_std"]), np.std(a, ddof=1), rtol=0, atol=1e-12)
    return {"shared_cases": len(expected), "matching_cases": len(expected), "designs_curves_and_summaries_match": True}


def verify_coefficient_fea(evidence, registry):
    """Verify Fig. 5 from selected actions through solver responses to mean/SD."""
    evidence = Path(evidence)
    expected = {r["case_id"]: r for r in coefficient_cases(evidence, registry)}
    sweeps = json.loads(Path(registry).read_text(encoding="utf-8"))
    registered_requests = {r["request_id"] for r in json.loads(
        Path(registry).with_name("fea_requests.json").read_text(encoding="utf-8"))}
    metrics = {}
    solver_hashes, releases = set(), set()
    with zipfile.ZipFile(evidence/"coefficients.zip") as coef, \
            zipfile.ZipFile(evidence/"policies.zip") as policies, \
            zipfile.ZipFile(evidence/"fea.zip") as fea:
        document = read_json(coef, "results/FEA_records.json")
        assert document["response_source"] == "FEA"
        records = document["records"]
        links = read_json(coef, "FEA_LINKS.json")
        assert len(records) == len(links) == 135
        records = {r["case_id"]: r for r in records}
        links = {r["case_id"]: r for r in links}
        assert set(records) == set(links) == set(expected), "incomplete coefficient FEA coverage"
        requests = {r["request_id"]: r for r in read_json(fea, "linkage.json")}
        for cid, case in expected.items():
            r, link = records[cid], links[cid]
            goal = case["goal"]
            for key in ["goal", "seed", "C1", "C2"]:
                assert r[key] == case[key], (cid, key)
            request = cid + f"/fea/g{goal}"
            assert request in registered_requests, f"unregistered FEA request: {request}"
            assert r["request_id"] == link["request_id"] == request
            assert link["fe_input_hash"] == r["fe_input_hash"] == requests[request]["fe_input_hash"]
            assert link["fea_member"] == r["fe_input_hash"] + ".json"
            fd = read_json(fea, link["fea_member"])
            assert fd["fe_input_hash"] == r["fe_input_hash"]
            assert fd["status"] == "success" and fd["physical_valid"]
            meta = fd["solver_meta"]
            assert meta["ansys_exit_code"] == 0 and meta["solve_converged"] == 1
            assert abs(meta["top_uy"] + 14.999) < 1e-8
            assert fd["solver_hash"] == r["solver_hash"]
            solver_hashes.add(fd["solver_hash"]); releases.add(meta["ansys_release"])
            selected = read_json(policies, cid + "/selected_designs.json")
            design, = selected["designs"]
            assert r["selected_episode"] == selected["selected_episode"]
            for key in ["actions30", "physical_thickness30"]:
                np.testing.assert_allclose(r[key], fd[key], rtol=0, atol=1e-12)
                np.testing.assert_allclose(fd[key], design[key], rtol=0, atol=1e-7)
            np.testing.assert_array_equal(r["u9_mm"], fd["u9_mm"])
            u, a = np.asarray(fd["u9_mm"]), np.asarray(fd["actions30"])
            assert u.shape == (9,) and a.shape == (30,)
            assert np.isfinite(u).all() and np.isfinite(a).all()
            np.testing.assert_allclose(fd["physical_thickness30"], 1.2 + .3*(a+1), rtol=0, atol=1e-7)
            i = goal - 1
            rho = float(np.mean((a+1)/2))
            weight = 1.0 if case["C2"] == 0 else .5 + case["C2"]*rho
            values = {
                "kappa": float(u[i]/(.5*(u[i-1]+u[i+1]))),
                "normalized_mean_thickness": rho,
                "bilateral_margin_mm": float(u[i]-max(u[i-1],u[i+1])),
                "global_margin_mm": float(u[i]-np.max(np.delete(u,i))),
                "coefficient_reward": float((u[i]-case["C1"]*(u[i-1]+u[i+1]))/weight),
                "canonical_reward": float((u[i]-.5*(u[i-1]+u[i+1]))/(.5+rho)),
            }
            for key, value in values.items():
                np.testing.assert_allclose(r[key], value, rtol=0, atol=1e-12, err_msg=f"{cid}: {key}")
            metrics[cid] = values
        assert len(solver_hashes) == 1 and releases == {"v231"}
        checked = 0
        for stage, panel in [("C1", "a"), ("C2", "b")]:
            data = coef.read(f"results/{stage}_summary.csv")
            assert data == (evidence/f"fig5/Fig5{panel}_{stage}_mean_sd.csv").read_bytes(), "figure CSV differs from FEA summary"
            rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
            expected_groups = {(r["goal"], r["C1"], r["C2"]) for r in sweeps[stage]}
            assert len(rows) == len(expected_groups) == 15
            assert {(int(r["goal"]), float(r["C1"]), float(r["C2"])) for r in rows} == expected_groups
            for row in rows:
                assert row["stage"] == stage and row["response_source"] == "FEA"
                cases = [c for c in sweeps[stage] if (c["goal"],c["C1"],c["C2"]) ==
                         (int(row["goal"]),float(row["C1"]),float(row["C2"]))]
                assert len(cases) == int(row["n"]) == 5
                for key in metrics[cases[0]["case_id"]]:
                    values = [metrics[c["case_id"]][key] for c in cases]
                    np.testing.assert_allclose(float(row[key+"_mean"]), np.mean(values), rtol=0, atol=1e-12)
                    np.testing.assert_allclose(float(row[key+"_sd"]), np.std(values,ddof=1), rtol=0, atol=1e-12)
                checked += 1
    return {"fea_cases":len(metrics), "condition_rows":checked, "response_source":"FEA",
            "sample_sd_ddof":1, "selected_designs_solver_responses_and_plot_tables_match":True}


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
    for name in ["coefficients.zip", "policies.zip", "fea.zip", "bilateral.zip"]:
        assert hashlib.sha256((args.evidence/name).read_bytes()).hexdigest() == index["archives"][name]["sha256"], name
        with zipfile.ZipFile(args.evidence/name) as archive:
            for member, meta in read_json(archive, "MANIFEST.json").items():
                data = archive.read(member)
                assert hashlib.sha256(data).hexdigest() == meta["sha256"], member
                assert len(data) == meta["bytes"], member
    report = {"coefficient_policy_links": verify_coefficient_links(
        args.evidence, root/"reproduction/spec/coefficient_sweeps.json")}
    report["coefficient_fea"] = verify_coefficient_fea(
        args.evidence, root/"reproduction/spec/coefficient_sweeps.json")
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
