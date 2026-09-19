"""Regression tests for the publication reward and checkpoint interfaces."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cmame_rt.checkpoint_evaluation import evaluate_checkpoint
from cmame_rt.models_surrogate import CNN
from cmame_rt.protocol import case_by_id, load_reference_math
from cmame_rt.reward import (bilateral_reward, bilateral_reward_torch,
                             design_metrics, reward, reward_torch)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("c2", [0., .25, .5, .75, 1.])
def test_reward_matches_manuscript_for_all_coefficient_values(c2):
    # Deliberately asymmetric neighbors and nonzero reward catch the factor of two.
    u = np.array([1., 2., 3., 4., 6., 5., 3., 2., 1.])
    a = np.linspace(-1., 1., 30)
    denominator = 1. if c2 == 0 else .5 + c2*.5
    want = 1.5 / denominator
    assert reward(u, a, 5, C2=c2) == pytest.approx(want)
    assert design_metrics(u, a, 5, C2=c2)['reward'] == pytest.approx(want)
    assert load_reference_math().design_metrics(u, a, 5, C2=c2)['reward'] == pytest.approx(want)
    got = reward_torch(torch.tensor(u[None]), torch.tensor(a[None]), 5, C2=c2)
    assert got.item() == pytest.approx(want, rel=1e-6)


def test_checkpoint_evaluator_distinguishes_best_from_last(tmp_path):
    case = case_by_id('surrogate/cnn/source/N30000/scratch/s0')
    (tmp_path/'resolved_config.json').write_text(json.dumps({'case':case}))
    (tmp_path/'train_log.csv').write_text('epoch\n1\n2\n')
    (tmp_path/'metrics.json').write_text(json.dumps({'best_epoch':1}))
    (tmp_path/'scaler.json').write_text(json.dumps({'mean':[0.]*9,'scale':[1.]*9}))
    model = CNN()
    torch.save(model.state_dict(),tmp_path/'best.pt')
    with torch.no_grad():
        model.fc[4].bias.add_(1.)
    torch.save(model.state_dict(),tmp_path/'last.pt')
    best = evaluate_checkpoint(tmp_path,'best',device='cpu')
    last = evaluate_checkpoint(tmp_path,'last',device='cpu')
    assert (best['epoch'],last['epoch']) == (1,2)
    assert not best['is_epoch_300'] and not last['is_epoch_300']
    assert best['checkpoint_sha256'] != last['checkpoint_sha256']
    assert json.loads((tmp_path/'metrics.json').read_text()) == {'best_epoch':1}
    with np.load(tmp_path/'predictions_common_test_best.npz') as b, np.load(tmp_path/'predictions_common_test_last.npz') as l:
        np.testing.assert_allclose(l['pred_mm']-b['pred_mm'],1.,atol=2e-7)
        assert np.array_equal(b['record_id'],l['record_id'])
    (tmp_path/'train_log.csv').write_text('epoch\n1\n300\n')
    with pytest.raises(ValueError,match='nonconsecutive'):
        evaluate_checkpoint(tmp_path,'last')


def test_coefficient_sweeps_use_common_cases_and_reuse_identical_pairs():
    sweeps = json.loads((ROOT/'spec/coefficient_sweeps.json').read_text())
    assert {k:len(v) for k,v in sweeps.items()} == {'C1':75,'C2':75,'joint':60}
    assert len({r['case_id'] for r in sweeps['C1']+sweeps['C2']}) == 135
    for stage,rows in sweeps.items():
        for r in rows:
            c = case_by_id(r['case_id'])
            assert c['algorithm']=='SAC' and c['domain']=='source' and c['arm']=='P0'
            assert c['C1']==r['C1'] and c['C2']==r['C2']
            assert c['traversal']=='rowwise_raster'
            assert c['dependencies']==[f"surrogate/cnn/source/N30000/scratch/s{r['seed']}"]
    # The final C2 pair uses the same source policies as the main experiment.
    final = [r for r in sweeps['C2'] if r['C2']==1.]
    assert all('/P0/' in r['case_id'] for r in final)


def test_bilateral_reward_is_separate_from_arithmetic_regression():
    u = np.array([0., 1., 2., 4., 8., 7., 3., 2., 1.])
    a = np.linspace(-1., 1., 30)
    arithmetic = reward(u, a, 5)
    bilateral = bilateral_reward(u, a, 5)
    assert arithmetic == pytest.approx(2.5 / 1.)
    assert bilateral == pytest.approx(1. / 1.)
    ut = torch.tensor(u[None], dtype=torch.float32)
    at = torch.tensor(a[None], dtype=torch.float32)
    assert reward_torch(ut, at, 5).item() == pytest.approx(arithmetic)
    assert bilateral_reward_torch(ut, at, 5).item() == pytest.approx(bilateral)


def test_s10_registry_uses_common_matched_dependencies():
    import importlib.util

    path = ROOT / "studies" / "bilateral" / "run.py"
    spec = importlib.util.spec_from_file_location("s10_bilateral_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.cases()
    assert len(rows) == 105
    assert sum(c["kind"] == "policy" and c["domain"] == "source"
               for c in rows) == 30
    assert sum(c["kind"] == "policy" and c["domain"] == "upper"
               for c in rows) == 60
    assert sum(c["kind"] == "optimizer" for c in rows) == 15
    table = {c["case_id"]: c for c in rows}
    for algorithm in ("SAC", "DDPG"):
        for goal in (5, 6, 7):
            for seed in range(5):
                target = table[f"upper/FGL{goal}/{algorithm}-TRL/s{seed}"]
                assert target["objective"] == "bilateral"
                assert target["dependencies"][1] == (
                    f"source/FGL{goal}/{algorithm}-SOURCE/s{seed}")
                assert target["dependencies"][0] == (
                    f"surrogate/cnn/upper/N6000/transfer/s{seed}")
