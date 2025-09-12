#!/usr/bin/env python
"""Generate markdown comparing original fusion vs fusion_calibrated and per-head raw vs chosen calibrator metrics."""
import os, json

HEAD_DIR='artifacts/diagnostics/head_calibration'
DIAG_DIR='artifacts/diagnostics'
FUSION_BASE='artifacts/fusion/calibration.json'
FUSION_CAL='artifacts/fusion_calibrated/metrics.json'
FUSION_CAL_IDS='artifacts/fusion_calibrated_ids/metrics.json'
META_CAL='artifacts/meta_fusion_calibrated/metrics.json'
OUT=os.path.join(DIAG_DIR,'calibration_diff.md')

def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)

def main():
    os.makedirs(DIAG_DIR, exist_ok=True)
    base=load_json(FUSION_BASE)
    cal=load_json(FUSION_CAL)
    cal_ids=load_json(FUSION_CAL_IDS)
    meta=load_json(META_CAL)
    lines=["# Calibration Diff",""]
    if base and cal:
        base_pr=base['metrics'].get('pr_auc') if 'metrics' in base else base.get('pr_auc')
        base_roc=base['metrics'].get('roc_auc') if 'metrics' in base else base.get('roc_auc')
        cal_val=cal['val']; cal_test=cal['test']
        lines.append("## Fusion Model")
        lines.append("")
        lines.append("Metric | Fusion Val | FusionCal Val | ΔVal | FusionCal Test")
        lines.append("------ | ---------- | ------------- | ---- | --------------")
        if base_pr is not None:
            lines.append(f"PR AUC | {base_pr:.4f} | {cal_val['pr_auc']:.4f} | {cal_val['pr_auc']-base_pr:+.4f} | {cal_test['pr_auc']:.4f}")
        if base_roc is not None:
            lines.append(f"ROC AUC | {base_roc:.4f} | {cal_val['roc_auc']:.4f} | {cal_val['roc_auc']-base_roc:+.4f} | {cal_test['roc_auc']:.4f}")
        lines.append(f"Brier | n/a | {cal_val['brier']:.6f} | n/a | {cal_test['brier']:.6f}")
        lines.append(f"ECE | n/a | {cal_val['ece']:.4f} | n/a | {cal_test['ece']:.4f}")
        lines.append("")
    if cal_ids:
        lines.append("## Fusion Calibrated (ID Join) Operating Points")
        opv=cal_ids.get('operating_point_val',{})
        opt=cal_ids.get('operating_point_test',{})
        lines.append("Split | Threshold | Precision | Recall | FPR | PR AUC | ROC AUC | Brier | ECE")
        lines.append("----- | --------- | --------- | ------ | --- | ------ | ------- | ------ | ---")
        lines.append(f"val | {opv.get('threshold',float('nan')):.6f} | {opv.get('precision',float('nan')):.4f} | {opv.get('recall',float('nan')):.4f} | {opv.get('fpr',float('nan')):.4f} | {cal_ids['val']['pr_auc']:.4f} | {cal_ids['val']['roc_auc']:.4f} | {cal_ids['val']['brier']:.6f} | {cal_ids['val']['ece']:.4f}")
        lines.append(f"test | {opt.get('threshold',float('nan')):.6f} | {opt.get('precision',float('nan')):.4f} | {opt.get('recall',float('nan')):.4f} | {opt.get('fpr',float('nan')):.4f} | {cal_ids['test']['pr_auc']:.4f} | {cal_ids['test']['roc_auc']:.4f} | {cal_ids['test']['brier']:.6f} | {cal_ids['test']['ece']:.4f}")
        lines.append("")
    if meta:
        lines.append("## Meta Fusion Calibrated Operating Points")
        opv=meta.get('operating_point_val',{})
        opt=meta.get('operating_point_test',{})
        lines.append("Split | Threshold | Precision | Recall | FPR | PR AUC | ROC AUC | Brier | ECE")
        lines.append("----- | --------- | --------- | ------ | --- | ------ | ------- | ------ | ---")
        lines.append(f"val | {opv.get('threshold',float('nan')):.6f} | {opv.get('precision',float('nan')):.4f} | {opv.get('recall',float('nan')):.4f} | {opv.get('fpr',float('nan')):.4f} | {meta['val']['pr_auc']:.4f} | {meta['val']['roc_auc']:.4f} | {meta['val']['brier']:.6f} | {meta['val']['ece']:.4f}")
        lines.append(f"test | {opt.get('threshold',float('nan')):.6f} | {opt.get('precision',float('nan')):.4f} | {opt.get('recall',float('nan')):.4f} | {opt.get('fpr',float('nan')):.4f} | {meta['test']['pr_auc']:.4f} | {meta['test']['roc_auc']:.4f} | {meta['test']['brier']:.6f} | {meta['test']['ece']:.4f}")
        lines.append("")
    # Heads
    lines.append("## Per-Head Improvements (Validation)")
    lines.append("")
    lines.append("Head | Raw Brier | Chosen Brier | ΔBrier | Raw PR AUC | Chosen PR AUC | ΔPR AUC | Chosen")
    lines.append("---- | --------- | ------------ | ------ | ---------- | ------------- | -------- | ------")
    if os.path.isdir(HEAD_DIR):
        for fname in sorted(os.listdir(HEAD_DIR)):
            if not fname.endswith('.json'): continue
            js=load_json(os.path.join(HEAD_DIR,fname))
            if not js: continue
            head=js['head']
            val=js['val']
            raw=val['raw']; chosen=val['chosen_metrics']; chosen_name=val['chosen']
            lines.append(" | ".join([
                head,
                f"{raw['brier']:.6f}", f"{chosen['brier']:.6f}", f"{chosen['brier']-raw['brier']:+.6f}",
                f"{raw['pr_auc']:.4f}", f"{chosen['pr_auc']:.4f}", f"{chosen['pr_auc']-raw['pr_auc']:+.4f}",
                chosen_name
            ]))
    with open(OUT,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+"\n")
    print('Wrote', OUT)

if __name__=='__main__':
    main()
