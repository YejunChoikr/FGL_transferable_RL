"""Frozen-data, transfer, reporting and standalone-runner regression tests."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from cmame_rt import data, encoding, models_surrogate as ms, surrogate_train as st
from cmame_rt.protocol import (PROTOCOL_HASH, canonical_digest, hash_report,
                               case_by_id, load_cases, load_reference_math)
from cmame_rt.reporting import training_summary, aggregate_curves


def test_input_checksums():
    assert data.verify_data_lock()["ok"]


def test_protocol_and_registry_hashes():
    assert hash_report()["protocol_hash_matches_spec"]
    assert len(load_cases()) == 695
    for case in load_cases():
        assert case["protocol_hash"] == PROTOCOL_HASH
        assert case["case_hash"] == canonical_digest({k:v for k,v in case.items() if k != "case_hash"})


@pytest.mark.parametrize("domain", data.DOMAINS)
def test_cached_encoding_matches_policy_and_reference(domain):
    d = data.load_domain(domain)
    # All cached records use the identical encoding consumed by the policy.
    assert np.array_equal(d.X60, encoding.encode_batch_numpy(d.A30))
    subset = d.A30[::97]
    golden = np.array([load_reference_math().encode_design(a) for a in subset])
    assert np.array_equal(golden, encoding.encode_batch_torch(torch.from_numpy(subset)).numpy())


@pytest.mark.parametrize("domain", data.DOMAINS)
def test_budget_nesting_common_test_and_training_only_scalers(domain):
    d = data.load_domain(domain)
    previous = {k: set() for k in ("train_idx", "val_idx", "test_idx")}
    common = None
    for n in data.BUDGETS[domain]:
        split = data.budget_split(domain, n)
        train, val, test = [set(split[k]) for k in previous]
        assert (len(train), len(val), len(test)) == (n*8//10, n//10, n//10)
        assert not train & val and not train & test and not val & test
        assert not (train | val) & set(split["common_test_idx"])
        for k in previous:
            assert previous[k] <= set(split[k])
            previous[k] = set(split[k])
        if common is not None:
            assert np.array_equal(common, split["common_test_idx"])
        common = split["common_test_idx"]
        actual = data.fit_y_scaler(d.U9[split["train_idx"]])
        frozen = data.load_y_scaler(domain, n)
        np.testing.assert_allclose(actual["mean"], frozen["mean"], rtol=1e-12)
        np.testing.assert_allclose(actual["scale"], frozen["scale"], rtol=1e-12)


def test_registered_surrogate_learning_rates():
    for c in load_cases():
        if c["kind"] == "supervised":
            assert st.expected_initial_lr(c) == (5e-5 if c["init"] == "transfer" else 5e-4)


def test_surrogate_transfer_resets_output_and_bn(monkeypatch, tmp_path):
    scratch = ms.CNN().state_dict()
    source = {k: torch.full_like(v, 3) for k, v in scratch.items()}
    source_path = tmp_path / "source.pt"
    torch.save(source, source_path)
    scratch_path = tmp_path / "scratch.pt"
    torch.save(scratch, scratch_path)
    monkeypatch.setattr(st, "load_surrogate_init", lambda *args: scratch)
    monkeypatch.setattr(st, "init_path", lambda *args: scratch_path)
    model, manifest = st.build_initial_model(
        case_by_id("surrogate/cnn/upper/N6000/transfer/s0"), source_path)
    state = model.state_dict()
    for k in ms.transfer_copy_keys("cnn", source):
        assert torch.equal(state[k], source[k])
    for k in ms.CNN_OUTPUT_KEYS:
        assert torch.equal(state[k], scratch[k])
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            assert torch.count_nonzero(layer.running_mean) == 0
            assert torch.equal(layer.running_var, torch.ones_like(layer.running_var))
            assert layer.num_batches_tracked == 0
    assert manifest["reset_output_equals_paired_scratch"]


def rows(n, shared=False, shift=0):
    return [dict(episode=i, goal=3+(i-1)%5 if shared else 5,
                 training_reward=i+shift) for i in range(1, n+1)]


def test_auc_uses_raw_first_300_and_curve_uses_trailing_mean():
    s = training_summary(rows(400), [5])["goals"]["5"]
    assert s["early_auc"] == 150.5
    assert s["average_reward"][0] == 1
    assert s["average_reward"][49] == 25.5
    assert s["average_reward"][50] == 26.5
    assert training_summary(rows(25), [5])["goals"]["5"]["early_auc"] is None


def test_shared_auc_uses_common_episode_window():
    s = training_summary(rows(1500, True), [3, 4, 5, 6, 7])
    for g in range(3, 8):
        v = s["goals"][str(g)]
        assert v["early_observation_count"] == 60
        assert v["early_auc"] == np.mean(np.arange(g-2, 301, 5))


def test_zero_based_runtime_log_matches_one_based_episode_counts():
    one_based = rows(1500, True)
    zero_based = [dict(r, episode=r["episode"]-1) for r in one_based]
    assert training_summary(zero_based, [3,4,5,6,7]) == training_summary(one_based, [3,4,5,6,7])


def test_curves_report_sample_sd():
    result = aggregate_curves([training_summary(rows(300), [5]),
        training_summary(rows(300, shift=2), [5])], 5)
    assert result["mean"][0] == 2
    np.testing.assert_allclose(result["sd"], np.sqrt(2))


def test_direct_dependencies_use_supplied_time_cap():
    ids = run.plan("optimizer/LBFGSB/upper/g5/s0")
    assert ids[-1] == "optimizer/LBFGSB/upper/g5/s0"
    assert ids[0] == "surrogate/cnn/source/N30000/scratch/s0"
    assert not any(c.startswith("policy/") for c in ids)


def test_completed_artifact_tampering_is_detected(tmp_path):
    f = tmp_path / "result.json"
    f.write_text("{}")
    marker = dict(case_id="test", protocol_hash=run.PROTOCOL_HASH,
                  smoke=True, code_sha256=run.code_digest(),
                  artifact_sha256={f.name: run.sha256(f)})
    (tmp_path / "completed.json").write_text(json.dumps(marker))
    assert run.validate_completed(tmp_path, "test", True)
    f.write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match="modified artifact"):
        run.validate_completed(tmp_path, "test", True)
