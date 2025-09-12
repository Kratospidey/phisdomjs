#!/usr/bin/env python
"""Cluster false positives to surface hard-negative themes.

Approach:
 1. Load fusion_calibrated_ids test predictions (or specified file).
 2. Identify false positives above chosen operating threshold.
 3. Build lightweight token features from URL/text heads (if available) by joining with head predictions (id->prob not used) plus raw data if present (pages_test*.jsonl lookup optional).
 4. Simple TF-IDF (manual) limited vocabulary top-N tokens.
 5. K-means (random init, few iterations).
 6. Output per-cluster: size, top tokens, sample IDs.

Outputs:
  artifacts/diagnostics/fp_clusters.json
  artifacts/diagnostics/fp_clusters.md

Note: Minimal dependencies (pure Python + numpy). If text source file not present, clusters only on URL tokenization using id mapping via url_head preds.
"""
from __future__ import annotations
import os, json, argparse, math, random, re
from collections import Counter, defaultdict
import numpy as np

FUSION_FILE='artifacts/fusion_calibrated_ids/preds_test.jsonl'
URL_HEAD_FILE='artifacts/url_head/preds_test.jsonl'
DATA_FILE_CANDIDATES=[
    'data/pages_test_full.jsonl',
    'data/pages_test.jsonl'
]
TOKEN_RE=re.compile(r"[A-Za-z0-9]{3,}")

def load_jsonl(path):
    if not os.path.isfile(path): return []
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            try: out.append(json.loads(ln))
            except Exception: pass
    return out

def index_by_id(recs):
    return {r.get('id'): r for r in recs if r.get('id')}

def tokenize_url(url: str):
    return TOKEN_RE.findall(url.lower()) if url else []

def build_text_map():
    for cand in DATA_FILE_CANDIDATES:
        if os.path.isfile(cand):
            data=load_jsonl(cand)
            # Expect fields maybe: id, url, text (optional)
            return index_by_id(data)
    return {}

def tfidf_matrix(docs, max_vocab=500):
    # docs: list[list[str]]
    df=Counter()
    for toks in docs:
        for t in set(toks): df[t]+=1
    vocab=[w for w,_ in df.most_common(max_vocab)]
    idx={w:i for i,w in enumerate(vocab)}
    rows=len(docs); cols=len(vocab)
    mat=np.zeros((rows, cols), dtype=float)
    idf={w: math.log((1+rows)/(1+df[w]))+1 for w in vocab}
    for r,toks in enumerate(docs):
        cnt=Counter(toks)
        for w,c in cnt.items():
            j=idx.get(w); 
            if j is None: continue
            mat[r,j]= (c/len(toks))*idf[w]
    # L2 normalize
    norms=np.linalg.norm(mat, axis=1)+1e-9
    mat=mat/norms[:,None]
    return mat, vocab

def kmeans(X, k=8, iters=15, seed=42):
    random.seed(seed); np.random.seed(seed)
    n=X.shape[0]
    if n==0: return np.zeros(0,dtype=int), np.zeros((k,X.shape[1]))
    init_idx=np.random.choice(n, size=min(k,n), replace=False)
    centers=X[init_idx].copy()
    if centers.shape[0]<k:
        # pad
        extra=np.zeros((k-centers.shape[0], X.shape[1])); centers=np.vstack([centers, extra])
    labels=np.zeros(n,dtype=int)
    for _ in range(iters):
        dists = ((X[:,None,:]-centers[None,:,:])**2).sum(axis=2)
        labels = dists.argmin(axis=1)
        for ci in range(k):
            mask=labels==ci
            if np.any(mask): centers[ci]=X[mask].mean(axis=0)
    return labels, centers

def top_terms_for_cluster(X, labels, vocab, cluster, topn=10):
    mask=labels==cluster
    if not np.any(mask): return []
    mean_vec=X[mask].mean(axis=0)
    idxs=np.argsort(-mean_vec)[:topn]
    return [vocab[i] for i in idxs if mean_vec[i]>0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fusion-file', default=FUSION_FILE)
    ap.add_argument('--url-head-file', default=URL_HEAD_FILE)
    ap.add_argument('--threshold', type=float, default=None, help='Override threshold; else load from operating_thresholds.json for fusion_calibrated_ids')
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--out-json', default='artifacts/diagnostics/fp_clusters.json')
    ap.add_argument('--out-md', default='artifacts/diagnostics/fp_clusters.md')
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    # Decide threshold
    if args.threshold is None:
        thresh=None
        oph='artifacts/diagnostics/operating_thresholds.json'
        if os.path.isfile(oph):
            js=json.load(open(oph,'r'))
            for fv in js.get('fusion_variants',[]):
                if fv['model']=='fusion_calibrated_ids': thresh=fv['threshold']
        if thresh is None: thresh=0.5
        args.threshold=thresh
    fusion=load_jsonl(args.fusion_file)
    data_map=build_text_map()
    url_map=index_by_id(load_jsonl(args.url_head_file))
    fps=[r for r in fusion if r.get('label')==0 and r.get('prob',0)>=args.threshold]
    docs=[]; ids=[]
    for r in fps:
        _id=r.get('id'); meta=data_map.get(_id) or {}
        url=meta.get('url') or url_map.get(_id,{}).get('url') or _id
        toks=tokenize_url(url)
        if not toks: continue
        docs.append(toks); ids.append(_id)
    X,vocab=tfidf_matrix(docs)
    labels, centers=kmeans(X, k=min(args.k, max(1,X.shape[0])))
    clusters=[]
    for ci in range(min(args.k, max(1,X.shape[0]))):
        mask=labels==ci
        if not np.any(mask): continue
        cluster_ids=[ids[i] for i in range(len(ids)) if labels[i]==ci]
        terms=top_terms_for_cluster(X, labels, vocab, ci)
        clusters.append({'cluster': ci,'size': len(cluster_ids),'top_terms': terms,'sample_ids': cluster_ids[:10]})
    # Sort by size
    clusters=sorted(clusters, key=lambda c:c['size'], reverse=True)
    with open(args.out_json,'w',encoding='utf-8') as f:
        json.dump({'threshold': args.threshold,'total_fp': len(fps),'clustered_fp': len(ids),'clusters': clusters}, f, indent=2)
    # Markdown
    lines=["# False Positive Clusters","",f"Threshold: {args.threshold:.6f}  Total FP: {len(fps)}  Clustered (non-empty URL tokens): {len(ids)}","", "Cluster | Size | Top Terms | Sample IDs","------- | ---- | --------- | ----------"]
    for c in clusters:
        lines.append(" | ".join([str(c['cluster']), str(c['size']), ",".join(c['top_terms'][:8]), ",".join(c['sample_ids'])]))
    with open(args.out_md,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+"\n")
    print('Wrote', args.out_md)

if __name__=='__main__':
    main()
