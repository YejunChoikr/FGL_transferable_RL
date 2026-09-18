"""Regression tests for the publication reward and checkpoint interfaces."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from cmame_rt.checkpoint_evaluation import evaluate_checkpoint
from cmame_rt.models_surrogate import CNN
from cmame_rt.protocol import case_by_id, load_reference_math
from cmame_rt.reward import reward, reward_torch, design_metrics

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


def test_bilateral_bo_routes_200_calls_to_bilateral_objective(tmp_path):
    # Exercise the real BO runner and surrogate. Replace only the GP search so
    # this checks objective wiring and the 200-call contract without a full fit.
    code = r'''
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0,sys.argv[1])
import run_bo
calls=[]
def search(objective,dimensions,**kw):
    assert len(dimensions)==30
    assert kw['n_calls']==200 and kw['n_initial_points']==20
    assert kw['acq_func']=='gp_hedge'
    for t in np.linspace(-1,1,200):
        calls.append(objective([t]*30))
    return SimpleNamespace(models=[])
run_bo.gp_minimize=search
out=Path(sys.argv[2]);run_bo.bo_run_dir=lambda *args:out
r=run_bo.run('FGL5',0,'local')
assert r['complete'] and r['n_objective_calls']==200
hist=json.loads((out/'objective_history.json').read_text())
for row,loss in zip(hist['records'],calls):
    u,a=row['u'],np.array(row['actions'])
    expected=min(u[4]-u[3],u[4]-u[5])/np.mean((a+2)/2)
    assert abs(loss+expected)<1e-5
'''
    result = subprocess.run([sys.executable,'-c',code,
        str(ROOT/'studies/bilateral/code'),str(tmp_path)],capture_output=True,text=True)
    assert result.returncode == 0,result.stdout+result.stderr
