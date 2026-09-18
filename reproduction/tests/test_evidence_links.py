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
    assert report["shared_cases"] == report["matching_cases"] == 30
    with zipfile.ZipFile(EVIDENCE/"bilateral.zip") as archive:
        records = links.read_json(archive, "solver_records.json")
        assert len(records) == 1
        links.verify_solver_record(archive, records[0])


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


@pytest.mark.parametrize("replacement", ["solve_converged=0", "top_uy=-10"])
def test_incomplete_solver_solution_is_rejected(replacement):
    text = "solve_converged=1\nr_force=-3.5\ntop_uy=-14.999\n"
    text += "\n".join(f"xr{i}={i}" for i in range(1,10))
    key = replacement.split("=")[0]
    text = "\n".join(replacement if line.startswith(key+"=") else line for line in text.splitlines())
    with pytest.raises(AssertionError):
        links.parse_solver_values(text)
