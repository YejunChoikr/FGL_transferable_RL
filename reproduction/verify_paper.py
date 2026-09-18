"""Recompute manuscript quantities from published run artifacts without training."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cmame_rt.reward import design_metrics, reward
from evidence_links import verify_coefficient_links, verify_coefficient_fea, verify_solver_record


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def j(z,name): return json.loads(z.read(name))
def stats(values):
    return {"mean":float(np.mean(values)),"sd":float(np.std(values,ddof=1)),"n":len(values)}


def verify(evidence,output):
    index=json.loads((evidence/"INDEX.json").read_text(encoding="utf-8"))
    report={"archive_integrity":{},"surrogate_transfer":{},"source_epoch300":{},"shared_peak_success":{}}
    for name,record in index['archives'].items():
        assert sha(evidence/name)==record['sha256'],name
        with zipfile.ZipFile(evidence/name) as z:
            for member,meta in j(z,"MANIFEST.json").items():
                assert hashlib.sha256(z.read(member)).hexdigest()==meta['sha256'],(name,member)
        report['archive_integrity'][name]=True
    with zipfile.ZipFile(evidence/'surrogates.zip') as z:
        for domain in ['upper','lower','ar08']:
            means={}
            for arm in ['scratch','transfer']:
                vals=[]
                for seed in range(5):
                    path=f'surrogate/cnn/{domain}/N6000/{arm}/s{seed}/predictions_common_test.npz'
                    with np.load(io.BytesIO(z.read(path)),allow_pickle=False) as d:
                        assert len(d['record_id'])==600
                        vals.append(float(np.abs(d['pred_mm']-d['true_mm']).mean()))
                means[arm]=stats(vals)
            means['mae_reduction_percent']=100*(1-means['transfer']['mean']/means['scratch']['mean'])
            report['surrogate_transfer'][domain]=means
        for arch in ['cnn','mlp']:
            values={k:[] for k in ['mae_mm','rmse_mm','mean_probe_r2']}
            for seed in range(5):
                base=f'surrogate/{arch}/source/N30000/scratch/s{seed}'
                log=list(csv.DictReader(io.StringIO(z.read(base+'/train_log.csv').decode())))
                assert int(log[-1]['epoch'])==300
                with np.load(io.BytesIO(z.read(base+'/predictions_common_test_last.npz')),allow_pickle=False) as d:
                    err=d['pred_mm']-d['true_mm']
                    values['mae_mm'].append(float(np.abs(err).mean()))
                    values['rmse_mm'].append(float(np.sqrt(np.mean(err**2))))
                    r2=1-(err**2).sum(0)/((d['true_mm']-d['true_mm'].mean(0))**2).sum(0)
                    values['mean_probe_r2'].append(float(r2.mean()))
            report['source_epoch300'][arch]={k:stats(v) for k,v in values.items()}
    policy_rows=[]
    with zipfile.ZipFile(evidence/'policies.zip') as pol,zipfile.ZipFile(evidence/'fea.zip') as fea:
        links={r['request_id']:r for r in j(fea,'linkage.json')}
        for name in pol.namelist():
            if not name.endswith('/selected_designs.json'): continue
            cid=name.rsplit('/',1)[0];case=j(pol,cid+'/case.json')
            selection=j(pol,cid+'/selection.json')
            assert selection['selected_episode']==selection['episodes'][int(np.argmax(selection['mean_scores']))]
            designs=j(pol,name)
            assert designs['selected_episode']==selection['selected_episode']
            for design in designs['designs']:
                goal=design['goal'];rid=f'{cid}/fea/g{goal}'
                if rid not in links: continue
                link=links[rid];fd=j(fea,link['fe_input_hash']+'.json')
                np.testing.assert_allclose(fd['actions30'],design['actions30'],rtol=0,atol=1e-7)
                np.testing.assert_allclose(fd['physical_thickness30'],design['physical_thickness30'],rtol=0,atol=1e-7)
                m=design_metrics(fd['u9_mm'],fd['actions30'],goal)
                policy_rows.append({'case_id':cid,'domain':case['domain'],'seed':case['seed'],
                    'goal':goal,'selected_episode':selection['selected_episode'],
                    'fea_input_hash':link['fe_input_hash'],**m})
        for domain in ['upper','lower','ar08']:
            rows=[x for x in policy_rows if x['case_id'].startswith(f'policy/SAC/{domain}/shared/ST_PAC/')]
            assert len(rows)==25,(domain,len(rows))
            report['shared_peak_success'][domain]={'successes':sum(x['peak_success'] for x in rows),'total':len(rows)}
    with zipfile.ZipFile(evidence/'bilateral.zip') as z:
        rows=j(z,'final_upper_runs.json')
        c4_fea=j(z,'ddpg_c4_fea.json')
        solver_records=j(z,'solver_records.json')
        solver_profiles={(r['task'],r['method'],r['seed']):verify_solver_record(z,r)
                         for r in solver_records}
        report['solver_evidence']={'verified_records':len(solver_profiles),
                                  'selected_design_and_nodal_output_match':True}
        report['bilateral']={}
        for task in ['FGL5','FGL6','FGL7']:
            for method in ['SAC-RL','SAC-TRL','DDPG-RL','DDPG-TRL','BO']:
                group=[r for r in rows if r['domain']=='upper' and r['task']==task and r['method']==method]
                assert len(group)==5
                values={'margin_mm':[],'arithmetic_reward':[]}
                unavailable=[]
                for r in group:
                    if not r['fea_u1']:
                        unavailable.append({'seed':int(r['seed']), 'status':r['fea_status'],
                                            'diagnostic':r.get('fea_failure_diagnostic','')})
                        continue
                    selected=j(z, f"{'bo_runs' if method=='BO' else 'runs'}/upper/{task}/{method}/seed_{r['seed']}/best_design.json")
                    a=np.array([float(r[f'action_{i:02d}']) for i in range(1,31)])
                    np.testing.assert_allclose(a,selected['actions'],rtol=0,atol=1e-7)
                    u=np.array([float(r[f'fea_u{i}']) for i in range(1,10)])
                    if method=='DDPG-TRL':
                        raw=next(x for x in c4_fea if x['task']==task and x['seed']==r['seed'])
                        if raw['status']=='success':
                            raw_u=[float(raw[f'u{i}']) for i in range(1,10)]
                        else:
                            raise ValueError(f"missing successful FEA record: {task}/{method}/{r['seed']}")
                        key=(task,method,int(r['seed']))
                        if raw.get('solver_record_member'):
                            assert key in solver_profiles, f"missing solver evidence: {key}"
                        if key in solver_profiles:
                            np.testing.assert_array_equal(raw_u,solver_profiles[key])
                    else:
                        raw_rows=list(csv.DictReader(io.StringIO(z.read(
                            f'ansys/upper_{task}/outputs/fea_validation_summary.csv').decode('utf-8-sig'))))
                        raw=next(x for x in raw_rows if x['method'].upper()==method and x['seed']==r['seed'])
                        raw_u=[float(raw[f'fea_u{i}']) for i in range(1,10)]
                        np.testing.assert_allclose(a,[float(raw[f'action_{i:02d}']) for i in range(1,31)],rtol=0,atol=1e-7)
                        assert raw['status']=='success'
                    np.testing.assert_allclose(u,raw_u,rtol=0,atol=1e-7)
                    m=design_metrics(u,a,int(task[-1]))
                    np.testing.assert_allclose(m['reward'],float(r['fea_original_reward']),rtol=0,atol=1e-6)
                    values['margin_mm'].append(m['bilateral_margin_mm']);values['arithmetic_reward'].append(m['reward'])
                report['bilateral'][task+'/'+method]={
                    **{k:stats(v) for k,v in values.items()},
                    'attempted':len(group), 'unavailable_fea':unavailable}
                assert not unavailable,(task,method,unavailable)
    report['arithmetic_objective_upper']={}
    with zipfile.ZipFile(evidence/'direct.zip') as direct,zipfile.ZipFile(evidence/'fea.zip') as fea:
        links={r['request_id']:r for r in j(fea,'linkage.json')}
        for goal in [5,6,7]:
            for method in ['SAC-RL','SAC-TRL','DDPG-RL','DDPG-TRL','BO']:
                if method=='BO':
                    selected=[]
                    for seed in range(5):
                        cid=f'optimizer/BO/upper/g{goal}/s{seed}'
                        candidate=j(direct,cid+'/incumbent.json')
                        fd=j(fea,links[cid+f'/fea/g{goal}']['fe_input_hash']+'.json')
                        np.testing.assert_allclose(fd['actions30'],candidate['actions30'],rtol=0,atol=1e-7)
                        selected.append(design_metrics(fd['u9_mm'],fd['actions30'],goal))
                else:
                    algo,arm=method.split('-');condition='ST_PAC' if arm=='TRL' else 'ST_P0'
                    prefix=f'policy/{algo}/upper/g{goal}/{condition}/N6000/'
                    selected=[x for x in policy_rows if x['case_id'].startswith(prefix)]
                assert len(selected)==5
                report['arithmetic_objective_upper'][f'FGL{goal}/{method}']={
                    'margin_mm':stats([x['bilateral_margin_mm'] for x in selected]),
                    'arithmetic_reward':stats([x['reward'] for x in selected])}
    assert sha(evidence/'table_s9.json')==index['table_s9_sha256']
    report['table_s9']=verify_s9(evidence/'table_s9.json')
    with zipfile.ZipFile(evidence/'coefficients.zip') as z:
        description=j(z,'DATA_DESCRIPTION.json')
        checked=0
        for name in z.namelist():
            if not name.endswith('/best_det_design.json'):
                continue
            match=re.search(r'/FGL(\d)/C1_([\d.]+)__C2_([\d.]+)/',name)
            goal,c1,c2=int(match[1]),float(match[2]),float(match[3])
            selected=j(z,name)
            actual=reward(selected['u'],selected['action'],goal,c1,c2)
            np.testing.assert_allclose(actual,selected['reward_env'],rtol=1e-6,atol=1e-6)
            with io.BytesIO(z.read(name.replace('best_det_design.json','reward_history.npy'))) as f:
                history=np.load(f,allow_pickle=False)
            assert len(history)==1500 and np.isfinite(history).all()
            checked+=1
        assert checked==135
        report['coefficient_figure_data']={'saved_runs':checked,**description}
    report['coefficient_policy_links']=verify_coefficient_links(
        evidence, ROOT/'reproduction/spec/coefficient_sweeps.json')
    report['coefficient_fea']=verify_coefficient_fea(
        evidence, ROOT/'reproduction/spec/coefficient_sweeps.json')
    # These are rounded manuscript values, checked only after independent aggregation.
    for domain,want in [('upper',35.3),('lower',29.8),('ar08',36.4)]:
        assert round(report['surrogate_transfer'][domain]['mae_reduction_percent'],1)==want
    for arch,want,sd in [('cnn',4.344,.159),('mlp',4.965,.602)]:
        r=report['source_epoch300'][arch]['mae_mm']
        assert round(r['mean']*1000,3)==want and round(r['sd']*1000,3)==sd
    assert report['shared_peak_success']['upper']['successes']==25
    output.mkdir(parents=True,exist_ok=True)
    (output/'verified_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (output/'selected_design_fea_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(policy_rows[0]));writer.writeheader();writer.writerows(policy_rows)
    print(json.dumps(report,indent=2))
    return report


def verify_s9(path):
    import torch
    from cmame_rt.models_surrogate import CNN
    from cmame_rt.encoding import encode_batch_numpy
    document=json.loads(Path(path).read_text(encoding='utf-8'));out=[]
    model_path=ROOT/document['designs'][0]['model']
    scaler_path=ROOT/document['designs'][0]['scaler']
    assert sha(model_path)==document['model_sha256']
    assert sha(scaler_path)==document['scaler_sha256']
    model=CNN()
    model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True));model.eval()
    scaler=json.loads(scaler_path.read_text(encoding='utf-8'))
    for r in document['designs']:
        x=encode_batch_numpy(np.array([r['actions30']]))
        with torch.no_grad():
            standardized=model(torch.tensor(x,dtype=torch.float32)).numpy().astype(np.float64)
            pred=(standardized*np.asarray(scaler['scale'])+np.asarray(scaler['mean']))[0]
        err=float(np.max(abs(pred-r['pred_mm'])))
        assert err<1e-5,(r['task'],err)
        out.append({'task':r['task'],'seed':r['seed'],'prediction_max_abs_difference_mm':err,
                    'model_sha256':sha(ROOT/r['model']),'scaler_sha256':sha(ROOT/r['scaler']),
                    'pred_mm':r['pred_mm'],'fea_mm':r['fea_mm'],
                    'error_percent':(100*np.abs(np.array(r['pred_mm'])-r['fea_mm'])/np.abs(r['fea_mm'])).tolist()})
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,default=ROOT/'evidence')
    p.add_argument('--output',type=Path,default=ROOT/'reproduction/evidence_checks')
    args=p.parse_args();verify(args.evidence,args.output)
