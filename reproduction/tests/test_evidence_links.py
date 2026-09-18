"""Saved-file consistency checks; no training, model inference or FEA execution."""
import json
from pathlib import Path
import zipfile

import pytest

import evidence_links as links

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/"evidence"
REGISTRY = ROOT/"reproduction/spec/coefficient_sweeps.json"


def test_supplied_coefficient_and_solver_records_match():
    report = links.verify_coefficient_links(EVIDENCE, REGISTRY)
    assert report["shared_cases"] == report["matching_cases"] == 135
    with zipfile.ZipFile(EVIDENCE/"bilateral.zip") as archive:
        records = links.read_json(archive, "solver_records.json")
        assert len(records) == 1
        links.verify_solver_record(archive, records[0])


def test_coefficient_fea_covers_every_design_and_matches_figure_tables():
    report = links.verify_coefficient_fea(EVIDENCE, REGISTRY)
    assert report["fea_cases"] == 135
    assert report["condition_rows"] == 30


@pytest.mark.parametrize("change", ["missing_case", "displacement", "summary"])
def test_inconsistent_coefficient_fea_is_rejected(monkeypatch, change):
    zip_class = zipfile.ZipFile

    class AlteredArchive:
        def __init__(self, path):
            self.archive = zip_class(path)
            self.coefficient = Path(path).name == "coefficients.zip"

        def __enter__(self): return self
        def __exit__(self, *args): self.archive.close()
        def namelist(self): return self.archive.namelist()

        def read(self, name):
            data = self.archive.read(name)
            if self.coefficient and name == "results/FEA_records.json":
                document = json.loads(data)
                if change == "missing_case": document["records"].pop()
                elif change == "displacement": document["records"][0]["u9_mm"][4] += .01
                return json.dumps(document).encode()
            if self.coefficient and name == "results/C1_summary.csv" and change == "summary":
                return data.replace(b"FEA", b"surrogate", 1)
            return data

    monkeypatch.setattr(links.zipfile, "ZipFile", AlteredArchive)
    with pytest.raises(AssertionError):
        links.verify_coefficient_fea(EVIDENCE, REGISTRY)


@pytest.mark.parametrize("field", ["action", "reward_env", "episode"])
def test_changed_coefficient_record_is_rejected(monkeypatch, field):
    zip_class = zipfile.ZipFile

    class AlteredArchive:
        def __init__(self, path):
            self.archive = zip_class(path)
            self.coefficient = Path(path).name == "coefficients.zip"

        def __enter__(self): return self
        def __exit__(self, *args): self.archive.close()
        def namelist(self): return self.archive.namelist()

        def read(self, name):
            data = self.archive.read(name)
            if self.coefficient and name.endswith("/best_det_design.json"):
                record = json.loads(data)
                if record.get("source_policy_case_id"):
                    if field == "action": record[field][0] += 0.001
                    else: record[field] += 1
                    return json.dumps(record).encode()
            return data

    monkeypatch.setattr(links.zipfile, "ZipFile", AlteredArchive)
    with pytest.raises(AssertionError):
        links.verify_coefficient_links(EVIDENCE, REGISTRY)


def test_missing_coefficient_policy_case_is_rejected(monkeypatch):
    zip_class = zipfile.ZipFile
    missing = "policy/SAC/source/g5/C1_0.1_C2_0/s0/selected_designs.json"

    class MissingPolicyMemberArchive:
        def __init__(self, path):
            self.archive = zip_class(path)
            self.policies = Path(path).name == "policies.zip"

        def __enter__(self): return self
        def __exit__(self, *args): self.archive.close()
        def namelist(self):
            return [name for name in self.archive.namelist()
                    if not (self.policies and name == missing)]
        def read(self, name): return self.archive.read(name)

    monkeypatch.setattr(links.zipfile, "ZipFile", MissingPolicyMemberArchive)
    with pytest.raises(AssertionError, match="missing sequential coefficient policy cases"):
        links.verify_coefficient_links(EVIDENCE, REGISTRY)


@pytest.mark.parametrize("replacement", ["solve_converged=0", "top_uy=-10"])
def test_incomplete_solver_solution_is_rejected(replacement):
    text = "solve_converged=1\nr_force=-3.5\ntop_uy=-14.999\n"
    text += "\n".join(f"xr{i}={i}" for i in range(1,10))
    key = replacement.split("=")[0]
    text = "\n".join(replacement if line.startswith(key+"=") else line for line in text.splitlines())
    with pytest.raises(AssertionError):
        links.parse_solver_values(text)
