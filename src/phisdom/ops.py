from __future__ import annotations
from typing import Sequence, Dict
import numpy as np

def brier(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    y=np.asarray(y_true, dtype=float); p=np.asarray(y_prob, dtype=float)
    if y.size==0: return float('nan')
    return float(np.mean((p - y)**2))

def ece(y_true: Sequence[int], y_prob: Sequence[float], bins: int = 10) -> float:
    y=np.asarray(y_true, dtype=float); p=np.asarray(y_prob, dtype=float)
    if y.size==0: return float('nan')
    edges=np.linspace(0,1,bins+1)
    idx=np.digitize(p, edges) - 1
    total=0.0
    for b in range(bins):
        m=idx==b
        if not np.any(m):
            continue
        conf=float(np.mean(p[m])); acc=float(np.mean(y[m])); w=float(np.mean(m))
        total += w * abs(conf-acc)
    return float(total)

def pick_operating_point(y_true: Sequence[int], y_prob: Sequence[float], target_recall: float = 0.95) -> Dict[str, float]:
    y=np.asarray(y_true, dtype=int); p=np.asarray(y_prob, dtype=float)
    if y.size==0:
        return {'threshold':1.0,'precision':float('nan'),'recall':float('nan'),'fpr':float('nan')}
    order=np.argsort(-p)
    y=y[order]; p=p[order]
    P=int(y.sum()); N=int(len(y)-P)
    tp=0; fp=0; best=None
    for score, label in zip(p,y):
        if label==1: tp+=1
        else: fp+=1
        recall=tp/max(1,P); precision=tp/max(1,tp+fp); fpr=fp/max(1,N)
        if recall>=target_recall:
            if best is None or precision > best['precision'] + 1e-12:
                best={'threshold': float(score),'precision': float(precision),'recall': float(recall),'fpr': float(fpr)}
    if best is None:
        precision=tp/max(1,tp+fp); recall=tp/max(1,P); fpr=fp/max(1,N)
        best={'threshold': float(p[-1]) if len(p) else 1.0,'precision': float(precision),'recall': float(recall),'fpr': float(fpr)}
    return best
