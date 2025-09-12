#!/usr/bin/env python
"""Enhanced FP clustering with richer feature extraction.

Adds to baseline clustering:
 - Domain token + path segment tokens
 - Character n-grams (3-5) of domain
 - Optional inclusion of a small sample of true negatives for contrast (--include-random-negatives)
 - Purity heuristics: fraction of tokens containing certain suspicious patterns (login, verify, secure, bank, update)

Outputs:
  artifacts/diagnostics/fp_clusters_enhanced.json
  artifacts/diagnostics/fp_clusters_enhanced.md

Relies only on built-in + numpy. Can be slower for very large sets; defaults keep vocab small.
"""
from __future__ import annotations
import os, json, argparse, random, re, math
from collections import Counter
import numpy as np

FUSION_FILE='artifacts/fusion_calibrated_ids/preds_test.jsonl'
URL_HEAD_FILE='artifacts/url_head/preds_test.jsonl'
DATA_FILES=[
    'data/pages_test_full.jsonl',
    'data/pages_test.jsonl'
]
TOKEN_RE=re.compile(r"[A-Za-z0-9]{3,}")
SUSPICIOUS={'login','verify','secure','bank','update','payment','account','wallet'}

def load_jsonl(path):
    if not os.path.isfile(path): return []
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            try: out.append(json.loads(ln))
            except Exception: pass
    return out

def index_by_id(rows):
    return {r.get('id'): r for r in rows if r.get('id')}

def split_url(url: str):
    if not url: return '', ''
    # crude split
    if '://' in url:
        url=url.split('://',1)[1]
    parts=url.split('/',1)
    domain=parts[0]
    path=parts[1] if len(parts)>1 else ''
    return domain, path

def domain_tokens(domain: str):
    return [p for p in re.split(r"[.\-]", domain.lower()) if p]

def path_tokens(path: str):
    # break on / ? & = - _ .
    return [p for p in re.split(r"[/?&=_.\-]", path.lower()) if p]

def char_ngrams(s: str, n_low=3, n_high=5):
    s=re.sub(r"[^a-z0-9]","", s.lower())
    grams=[]
    for n in range(n_low, n_high+1):
        for i in range(len(s)-n+1):
            grams.append(s[i:i+n])
    return grams

def build_docs(fps, url_map, data_map, include_random_negatives=False, neg_sample=100):
    docs=[]; ids=[]; meta=[]
    for r in fps:
        _id=r['id']
        url=url_map.get(_id,{}).get('url') or data_map.get(_id,{}).get('url') or _id
        d,p=split_url(url)
        toks=set()
        toks.update(domain_tokens(d))
        toks.update(path_tokens(p))
        toks.update(char_ngrams(d))
        toks.update(TOKEN_RE.findall(url.lower()))
        toks=[t for t in toks if len(t)<=20]
        if not toks: continue
        docs.append(list(toks)); ids.append(_id)
        meta.append({'id': _id, 'url': url})
    if include_random_negatives:
        negs=[r for r in url_map.values() if r.get('label')==0 and r.get('prob',0)<0.01]
        random.shuffle(negs)
        for r in negs[:neg_sample]:
            url=r.get('url') or r.get('id')
            d,p=split_url(url)
            toks=set(domain_tokens(d)+path_tokens(p)+char_ngrams(d)+TOKEN_RE.findall(url.lower()))
            if not toks: continue
            docs.append(list(toks)); ids.append(r.get('id'))
            meta.append({'id': r.get('id'), 'url': url, 'contrast': True})
    return docs, ids, meta

def tfidf(docs, max_vocab=1200):
    df=Counter()
    for toks in docs:
        for t in set(toks): df[t]+=1
    vocab=[w for w,_ in df.most_common(max_vocab)]
    idx={w:i for i,w in enumerate(vocab)}
    rows=len(docs); cols=len(vocab)
    X=np.zeros((rows, cols), dtype=float)
    idf={w: math.log((1+rows)/(1+df[w]))+1 for w in vocab}
    for r,toks in enumerate(docs):
        c=Counter(toks)
        for w,cnt in c.items():
            j=idx.get(w); 
            if j is None: continue
            X[r,j]=(cnt/len(toks))*idf[w]
    norms=np.linalg.norm(X,axis=1)+1e-12
    X=X/norms[:,None]
    return X, vocab

def kmeans(X, k=10, iters=25, seed=42):
    np.random.seed(seed)
    n=X.shape[0]
    if n==0: return np.zeros(0,dtype=int), np.zeros((k,X.shape[1]))
    init=np.random.choice(n, size=min(k,n), replace=False)
    centers=X[init].copy()
    if centers.shape[0]<k:
        centers=np.vstack([centers, np.zeros((k-centers.shape[0], X.shape[1]))])
    labels=np.zeros(n,dtype=int)
    for _ in range(iters):
        d=((X[:,None,:]-centers[None,:,:])**2).sum(axis=2)
        labels=d.argmin(axis=1)
        for ci in range(k):
            mask=labels==ci
            if np.any(mask): centers[ci]=X[mask].mean(axis=0)
    return labels, centers

def top_terms(X, labels, vocab, ci, topn=15):
    mask=labels==ci
    if not np.any(mask): return []
    mean_vec=X[mask].mean(axis=0)
    idxs=np.argsort(-mean_vec)[:topn]
    return [vocab[i] for i in idxs if mean_vec[i]>0]

def purity(tokens):
    if not tokens: return 0.0
    cnt=sum(1 for t in tokens if any(s in t for s in SUSPICIOUS))
    return cnt/len(tokens)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fusion-file', default=FUSION_FILE)
    ap.add_argument('--url-head-file', default=URL_HEAD_FILE)
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--include-random-negatives', action='store_true')
    ap.add_argument('--random-negatives', type=int, default=120)
    ap.add_argument('--max-vocab', type=int, default=1200)
    ap.add_argument('--out-json', default='artifacts/diagnostics/fp_clusters_enhanced.json')
    ap.add_argument('--out-md', default='artifacts/diagnostics/fp_clusters_enhanced.md')
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    # threshold
    if args.threshold is None:
        oph='artifacts/diagnostics/operating_thresholds.json'
        if os.path.isfile(oph):
            js=json.load(open(oph,'r'))
            for fv in js.get('fusion_variants',[]):
                if fv['model']=='fusion_calibrated_ids':
                    args.threshold=fv['threshold']
                    break
        if args.threshold is None: args.threshold=0.5
    fusion=load_jsonl(args.fusion_file)
    url_preds=load_jsonl(args.url_head_file)
    url_map=index_by_id(url_preds)
    # optional data file (for urls if needed)
    data_map={}
    for df in DATA_FILES:
        if os.path.isfile(df):
            for rec in load_jsonl(df):
                if rec.get('id') and rec.get('id') not in data_map:
                    data_map[rec['id']]=rec
    fps=[r for r in fusion if r.get('label')==0 and r.get('prob',0)>=args.threshold]
    docs, ids, meta = build_docs(fps, url_map, data_map, include_random_negatives=args.include_random_negatives, neg_sample=args.random_negatives)
    X,vocab=tfidf(docs, max_vocab=args.max_vocab)
    labels, centers=kmeans(X, k=min(args.k, max(1,X.shape[0])))
    clusters=[]
    for ci in range(min(args.k, max(1,X.shape[0]))):
        mask=labels==ci
        if not np.any(mask): continue
        cid_ids=[ids[i] for i in range(len(ids)) if labels[i]==ci]
        terms=top_terms(X, labels, vocab, ci)
        clusters.append({
            'cluster': ci,
            'size': len(cid_ids),
            'top_terms': terms,
            'purity_suspicious': purity(terms),
            'sample_ids': cid_ids[:15]
        })
    clusters=sorted(clusters, key=lambda c:c['size'], reverse=True)
    summary={'threshold': args.threshold,'total_fp': len(fps),'clustered_docs': len(ids),'clusters': clusters,'params': {'k': args.k,'max_vocab': args.max_vocab,'include_random_negatives': args.include_random_negatives}}
    with open(args.out_json,'w',encoding='utf-8') as f:
        json.dump(summary,f,indent=2)
    lines=['# Enhanced False Positive Clusters','',f'Threshold: {args.threshold:.6f}  Total FP: {len(fps)}  Docs clustered: {len(ids)}','', 'Cluster | Size | Purity(susp) | Top Terms | Sample IDs','------- | ---- | ------------- | --------- | ----------']
    for c in clusters:
        lines.append(' | '.join([str(c['cluster']), str(c['size']), f"{c['purity_suspicious']:.2f}", ','.join(c['top_terms'][:12]), ','.join(c['sample_ids'])]))
    with open(args.out_md,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    print('Wrote', args.out_md)

if __name__=='__main__':
    main()
