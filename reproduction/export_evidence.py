"""Export saved experimental artifacts; no training, inference or FEA is run."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean(value):
    """Retain experiment metadata while making source paths portable."""
    if isinstance(value, dict):
        return {k:clean(v) for k,v in value.items() if k not in
                {"host", "hostname", "executable", "owner", "machine_allocation"}}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str) and len(value)>2 and value[1:3] in (":/", ":\\"):
        value = value.replace("\\", "/")
        for anchor in ["/accepted/", "/assets/", "/runs/", "/source_bank/", "/scratch_bank/"]:
            if anchor in value:
                return value.split(anchor,1)[1] if anchor=="/accepted/" else anchor[1:]+value.split(anchor,1)[1]
        return value.rsplit("/",1)[-1]
    return value


class Bundle:
    def __init__(self, path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.zip=zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6)
        self.files={}

    def put(self,name, data, source_hash=None):
        name=str(name).replace("\\","/")
        if name in self.files:
            return
        if isinstance(data,str): data=data.encode("utf-8")
        entry=zipfile.ZipInfo(name,date_time=(2026,9,18,0,0,0))
        entry.compress_type=zipfile.ZIP_DEFLATED
        self.zip.writestr(entry,data)
        self.files[name]={"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
        if source_hash: self.files[name]["original_sha256"]=source_hash

    def json(self,name,obj,source_hash=None):
        self.put(name,json.dumps(clean(obj),ensure_ascii=False,separators=(",",":")),source_hash)

    def copy(self,path,name=None):
        path=Path(path); data=path.read_bytes(); h=hashlib.sha256(data).hexdigest()
        if path.suffix==".json": self.json(name or path.name,json.loads(data),h)
        elif path.suffix==".csv":
            rows=list(csv.reader(io.StringIO(data.decode('utf-8-sig'))))
            portable=[[clean(cell) for cell in row] for row in rows]
            if rows!=portable:
                buffer=io.StringIO(newline='');csv.writer(buffer).writerows(portable)
                data=buffer.getvalue().encode('utf-8')
            self.put(name or path.name,data,h)
        else: self.put(name or path.name,data,h)

    def close(self):
        self.json("MANIFEST.json",self.files)
        self.zip.close()
        return {"sha256":sha(self.path),"bytes":self.path.stat().st_size,"members":len(self.files)}


def export(unified,bilateral,ddpg_c4,coefficients,legacy,s9_fea,output):
    output.mkdir(parents=True,exist_ok=True)
    index={"schema":1,"source_studies":{
        "unified":unified.name,"bilateral":bilateral.name,"ddpg_c4":ddpg_c4.name,
        "coefficients":coefficients.name},
        "archives":{},"claims":{}}
    cases=read(unified/"spec/cases.json")
    sup=Bundle(output/"surrogates.zip")
    pol=Bundle(output/"policies.zip")
    direct=Bundle(output/"direct.zip")
    for case in cases:
        cid=case["case_id"];base=unified/"accepted"/cid
        bundle={"supervised":sup,"policy":pol,"optimizer":direct}[case["kind"]]
        bundle.json(cid+"/case.json",case)
        files=({"supervised":["metrics.json","predictions_common_test.npz","train_log.csv",
                              "lr_log.csv","scaler.json","best.pt","resolved_config.json"],
                "policy":["episode_log.csv","eval.jsonl","selection.json","selected_designs.json","resolved_config.json"],
                "optimizer":["incumbent.json","counts.json","timing.json","resolved_config.json"]}[case["kind"]])
        for name in files: bundle.copy(base/name,cid+"/"+name)
        if case["kind"]=="supervised" and case["domain"]=="source":
            bundle.copy(base/"last.pt",cid+"/last.pt")
            pred=unified/"figures_manuscript/derived"/f"source_last_epoch_predictions_{case['architecture']}_s{case['seed']}.npz"
            bundle.copy(pred,cid+"/predictions_common_test_last.npz")
        if case["kind"]=="optimizer":
            # Fig. 11 is limited to the first 1,500 design evaluations.
            with (base/"search_log.csv").open(encoding="utf-8",newline="") as f:
                reader=csv.reader(f); part=[]
                for i,row in enumerate(reader):
                    if i>1500: break
                    part.append(row)
            text=io.StringIO(newline="");csv.writer(text).writerows(part)
            bundle.put(cid+"/first1500.csv",text.getvalue())
    for b in [sup,pol,direct]: index["archives"][b.path.name]=b.close()
    fea=Bundle(output/"fea.zip")
    with (unified/"results/tables/fea_linkage.csv").open(encoding="utf-8-sig") as f:
        links=list(csv.DictReader(f))
    for row in links:
        h=row["fe_input_hash"];p=unified/"fea/results"/(h+".json")
        if p.is_file(): fea.copy(p,h+".json")
        else: raise FileNotFoundError(p)
    fea.json("linkage.json",links)
    index["archives"][fea.path.name]=fea.close()
    bil=Bundle(output/"bilateral.zip")
    for base in sorted((bilateral/"runs").glob("*/*/*/seed_*")):
        if base.parent.name == "DDPG-TRL":
            continue
        rel=base.relative_to(bilateral).as_posix()
        for name in ["best_design.json","completion.json","reward_history.npy","deterministic_history.json","initialization_audit.json"]:
            bil.copy(base/name,rel+"/"+name)
    for base in sorted((bilateral/"bo_runs/upper").glob("*/BO/seed_*")):
        rel=base.relative_to(bilateral).as_posix()
        for name in ["best_design.json","completion.json","objective_history.json"]:
            bil.copy(base/name,rel+"/"+name)
    for base in sorted((ddpg_c4/"runs/bilateral/upper").glob("*/seed_*")):
        if int(base.name.split('_')[-1]) not in range(5):
            continue
        rel=f"runs/upper/{base.parent.name}/DDPG-TRL/{base.name}"
        for name in ["best_design.json","completion.json","reward_history.npy",
                     "deterministic_history.json","transfer_tensor_audit.json"]:
            bil.copy(base/name,rel+"/"+name)
    with (legacy/"tables/optimal_models_all_seeds.csv").open(encoding="utf-8-sig") as f:
        final_rows=[x for x in csv.DictReader(f) if x["objective_family"]=="bilateral_reward"
                    and x["domain"]=="upper"]
    assert len(final_rows)==75
    # These are the final S10 cases: 60 original SAC/DDPG-RL/BO + 15 C4 DDPG-TRL.
    bil.json("final_upper_runs.json",final_rows,
             sha(legacy/"tables/optimal_models_all_seeds.csv"))
    recovery_script=legacy/"code/apply_recovered_failed_fea.py"
    tree=ast.parse(recovery_script.read_text(encoding='utf-8'))
    recovered=next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and
                   t.id=='RECOVERED' for t in n.targets))
    bil.json("recovered_fea_profiles.json",{
        "source_file":recovery_script.name,"source_sha256":sha(recovery_script),
        "provenance":"Recovered profiles recorded in the final-data-package correction; original solver output is not included for these entries.",
        "profiles":[x for x in recovered if x['objective_family']=='bilateral_reward']})
    for p in (bilateral/"ansys").rglob("*.csv"):
        bil.copy(p,p.relative_to(bilateral))
    for p in (ddpg_c4/"results").glob("fea_run_level.csv"):
        # CSV cells can contain source paths; publish portable JSON rows.
        with p.open(encoding="utf-8-sig") as f:
            rows=[r for r in csv.DictReader(f) if r.get("objective")=="bilateral"
                  and r.get("domain")=="upper" and int(r["seed"]) in range(5)]
        bil.json("ddpg_c4_fea.json",rows,sha(p))
    index["archives"][bil.path.name]=bil.close()
    coeff=Bundle(output/"coefficients.zip")
    for name in ["run_summary.csv","C1_summary.csv","C2_summary.csv","C1_selection.json","C2_selection.json","ansys_validation.csv"]:
        coeff.copy(coefficients/"results"/name,"results/"+name)
    for p in (coefficients/"runs").rglob("*"):
        if p.is_file() and p.name in ["reward_history.npy","best_det_design.json","deterministic_history.json","run_result.json","best_det.json"]:
            coeff.copy(p,p.relative_to(coefficients))
    index["archives"][coeff.path.name]=coeff.close()
    # The fabricated designs keep their original prediction model and full-precision inputs.
    table_path=legacy/"tables/optimal_models_best_predicted.csv"
    with table_path.open(encoding="utf-8-sig") as f:
        rows=[x for x in csv.DictReader(f) if x["domain"]=="upper" and
              x["method"]=="SAC-TRL" and x["objective_family"]=="original_reward"]
    s9=[]
    for x in rows:
        s9.append({"task":x["task"],"goal":int(x["task"][-1]),"seed":int(x["seed"]),
                   "model":"models/surrogate_upper.pth","scaler":"models/scaler_upper.json",
                   "actions30":[float(x[f"action_{i:02d}"]) for i in range(1,31)],
                   "pred_mm":[float(x[f"pred_u{i}"]) for i in range(1,10)],
                   "fea_mm":[float(x[f"fea_u{i}"]) for i in range(1,10)],
                   "action_sha256":x["action_sha256"],
                   "selection_rule":x["cross_seed_selection_rule"]})
    # Table S9 uses the measured-design FGL5 FEA profile retained in the revision audit.
    d=read(s9_fea)
    next(x for x in s9 if x['goal']==5)['fea_mm']=[x['fea_mm'] for x in d['rows']]
    repo=Path(__file__).resolve().parents[1]
    s9doc={"designs":s9,"source_table_sha256":sha(table_path),
           "fgl5_fea_audit_sha256":sha(s9_fea),
           "model_sha256":sha(repo/"models/surrogate_upper.pth"),
           "scaler_sha256":sha(repo/"models/scaler_upper.json"),
           "purpose":"Surrogate predictions used to select the fabricated designs; not the unified-rerun surrogate validation."}
    (output/"table_s9.json").write_text(json.dumps(s9doc,indent=2),encoding="utf-8")
    index["table_s9_sha256"]=sha(output/"table_s9.json")
    index["claims"]={"surrogate_transfer_mae":"surrogates.zip: 3 domains x scratch/transfer x five best.pt common-test predictions",
        "shared_upper_25_of_25":"policies.zip selected designs + fea.zip linkage by request_id and fe_input_hash",
        "Table_S6_Fig_S5":"surrogates.zip: source cnn/mlp, seeds 0-4, epoch-300 last.pt predictions",
        "Table_S9":"table_s9.json and original models/; current main Fig. 6 and experimental Fig. 10",
        "S10":"bilateral.zip and reproduction/studies/bilateral; arithmetic columns from unified policy/direct FEA",
        "Fig_5b":"coefficients.zip and reproduction/studies/reward_coefficients; C2=0 uses W=1",
        "Fig_11":"direct.zip first1500.csv; full time-cap selected candidate is in incumbent.json"}
    (output/"INDEX.json").write_text(json.dumps(index,indent=2),encoding="utf-8")
    print(json.dumps({k:round(v['bytes']/1e6,2) for k,v in index['archives'].items()},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["unified","bilateral","ddpg_c4","coefficients","legacy","s9_fea","output"]:
        p.add_argument("--"+name.replace('_','-'),type=Path,required=True)
    export(**vars(p.parse_args()))
