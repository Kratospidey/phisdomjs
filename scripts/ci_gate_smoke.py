#!/usr/bin/env python
"""CI gate negative-path smoke: perturb metrics to ensure gate fails.

Loads enhanced evaluation JSON, reduces precision by a delta over tolerance,
runs ci_gate.py and expects non-zero exit. Restores original file after.
"""
from __future__ import annotations
import os, json, subprocess, shutil, sys, tempfile

EVAL_FILE='artifacts/diagnostics/enhanced_evaluation.json'

def main():
    if not os.path.isfile(EVAL_FILE):
        sys.exit('Missing enhanced evaluation file; run evaluate_enhanced.py first.')
    data=json.load(open(EVAL_FILE,'r'))
    # Locate fusion op
    fusion=data.get('fusion_calibrated_ids') or {}
    op=fusion.get('operating_point_test') or fusion.get('test',{}).get('operating_point_test')
    if not op or not isinstance(op.get('precision'), (int,float)):
        sys.exit('Missing fusion operating precision; cannot run smoke')
    # Perturb: lower precision by 10%
    op['precision']=float(op['precision']) * 0.85
    # Write temp eval file
    tmp_dir=tempfile.mkdtemp()
    tmp_eval=os.path.join(tmp_dir,'enhanced_evaluation.json')
    with open(tmp_eval,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=2)
    # Run ci_gate against this temp file by temporarily swapping
    backup=EVAL_FILE+".bak"
    shutil.copy2(EVAL_FILE, backup)
    try:
        shutil.copy2(tmp_eval, EVAL_FILE)
        res=subprocess.run([sys.executable, 'scripts/ci_gate.py'], capture_output=True, text=True)
        print(res.stdout)
        if res.returncode == 0:
            print('[SMOKE][ERROR] CI gate passed unexpectedly after perturbation')
            sys.exit(7)
        else:
            print('[SMOKE] CI gate failed as expected (negative-path)')
    finally:
        shutil.copy2(backup, EVAL_FILE)
        os.remove(backup)
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__=='__main__':
    main()
