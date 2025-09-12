#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
import shutil
import html as _html
from typing import Any, Dict, List, Tuple, cast

import numpy as np

import torch
from transformers import AutoModelForSequenceClassification, MarkupLMProcessor, AutoTokenizer, T5EncoderModel
from transformers.utils import logging as hf_logging
import torch

# Safer defaults for heavy operations in constrained environments
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
except Exception:
    pass
try:
    # Limit CPU threads to reduce contention in WSL/low-RAM
    torch.set_num_threads(1)
except Exception:
    pass

from phisdom.data.loader import JsonlPhishDataset, MarkupLMDataCollator
from phisdom.data.js import concat_scripts

def escape_html(s: str) -> str:
    try:
        return _html.escape(str(s), quote=True)
    except Exception:
        return str(s)


def ensure_deps():
    missing = []
    try:
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
        from sklearn import metrics  # noqa: F401
    except Exception:
        missing.append("matplotlib seaborn scikit-learn")
    try:
        import lime  # noqa: F401
    except Exception:
        missing.append("lime")
    try:
        import shap  # noqa: F401
    except Exception:
        missing.append("shap")
    if missing:
        print("[INFO] Optional packages missing:", ", ".join(missing))
        print("[INFO] Install for full reports:")
        print("  pip install matplotlib seaborn scikit-learn lime shap")


def _ensure_text_str(x: Any, max_chars: int | None = None) -> str:
    """Coerce various text-like inputs (np.ndarray, bytes, lists) to a str.
    Truncates to max_chars if provided.
    """
    s: str
    try:
        import numpy as _np  # local import to avoid global shadowing
    except Exception:  # pragma: no cover
        _np = None  # type: ignore
    # Already str
    if isinstance(x, str):
        s = x
    # bytes -> utf-8
    elif isinstance(x, (bytes, bytearray)):
        s = bytes(x).decode("utf-8", errors="ignore")
    # numpy arrays
    elif _np is not None and isinstance(x, _np.ndarray):  # type: ignore[truthy-bool]
        if x.ndim == 0:
            s = str(x.item())
        else:
            try:
                s = " ".join(map(str, x.flatten().tolist()))
            except Exception:
                s = str(x)
    # lists/tuples -> join
    elif isinstance(x, (list, tuple)):
        try:
            s = " ".join(map(str, x))
        except Exception:
            s = str(x)
    else:
        s = str(x)
    if max_chars and len(s) > max_chars:
        s = s[:max_chars]
    return s


def _inject_link_into_html(file_path: str, url: str | None, extra_html: str = ""):
    """Prepend a small header with a link (if provided) and optional extra_html (e.g., a pill) into an HTML file."""
    try:
        link_html = f"<div>Source: <a href='{url}' target='_blank' style='color:#8ab4f8'>{url}</a></div>" if url else ""
        header = (
            "<div style=\"padding:8px 10px;margin:8px 0;border:1px solid #2a2f3a;"
            "background:#0b0e14;color:#cbd5e1;border-radius:6px;\">"
            f"<div style='display:flex;align-items:center;gap:10px'>{link_html}<div style='margin-left:auto'>{extra_html}</div></div>"
            "</div>"
        )
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        # Try to insert after <body> or at top
        if "<body" in html:
            # Find closing '>' of <body ...>
            import re
            m = re.search(r"<body[^>]*>", html, flags=re.IGNORECASE)
            if m:
                pos = m.end()
                html = html[:pos] + header + html[pos:]
            else:
                html = header + html
        else:
            html = header + html
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception:
        pass


def _safe_float(x: Any, default: float = float("nan")) -> float:
    """Best-effort float conversion that tolerates None/unknowns.
    Returns `default` (NaN by default) on failure.
    """
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _base_expl_css() -> str:
    return (
        "<style>"
        "body{background:#0f1115;color:#e6e6e6;font-family:system-ui,Arial,sans-serif;margin:0}"
        ".wrap{max-width:1200px;margin:0 auto;padding:16px}"
        ".top{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid #262b36;background:#0b0e14;position:sticky;top:0}"
        ".pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#1f2330;color:#cbd5e1;border:1px solid #2a2f3a;font-size:12px}"
        "a{color:#8ab4f8}"
        ".grid{display:grid;grid-template-columns:320px 1fr;gap:16px;margin-top:16px}"
        ".card{background:#0f1320;border:1px solid #262b36;border-radius:8px;padding:12px}"
        ".kv{display:grid;grid-template-columns:1fr auto;gap:6px 10px;font-size:14px} .kv div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
        ".tok{display:inline-block;margin:2px 2px;padding:2px 4px;border-radius:4px}"
        ".legend span{display:inline-block;padding:3px 8px;border-radius:999px;margin-right:8px;border:1px solid #333}"
        ".pos{background:rgba(235,245,255,0.3);color:#9cc4ff} .neg{background:rgba(255,235,235,0.3);color:#ff9c9c}"
        ".body{background:#0b0e14}"
        ".mono{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace}"
        ".tbl{width:100%;border-collapse:collapse} .tbl td,.tbl th{border:1px solid #333;padding:6px 8px} .tbl th{background:#0b0e14}"
        "</style>"
    )


def _wrap_expl_html(title: str, url: str | None, info: List[tuple[str, str]], body_html: str, details_extra_html: str = "", header_extra_html: str = "") -> str:
    css = _base_expl_css()
    info_rows = "".join(f"<div style='color:#9aa4b2'>{k}</div><div class='mono'>{v}</div>" for k, v in info)
    src_html = f"<a href='{url}' target='_blank'>{url}</a>" if url else ""
    header = (
        "<div class='top'>"
        f"<div class='pill'>Explain</div><div class='mono' style='opacity:.8'>{title}</div>"
        f"<div style='margin-left:auto'>{src_html}</div>"
        f"{header_extra_html}</div>"
    )
    return (
        "<html><head><meta charset='utf-8'><meta name='color-scheme' content='dark'>"
        f"<title>{title}</title>" + css + "</head><body class='body'>" + header +
        "<div class='wrap'><div class='grid'>"
        f"<div class='card'><h3 style='margin:0 0 8px 0'>Details</h3><div class='kv'>{info_rows}</div>" + details_extra_html + "</div>"
        f"<div class='card'>" + body_html + "</div>"
        "</div></div></body></html>"
    )


def _save_shap_text_html(out_path: str, text: str, tokens: List[str] | None, values: np.ndarray, class_index: int = 1, title: str = "SHAP Text Explanation", url: str | None = None, info: List[tuple[str,str]] | None = None, top_k: int = 15, details_extra_html: str = "", header_extra_html: str = ""):
        """Write a self-contained HTML showing token-level contributions with nice styling.
        Supports extra metadata (info) and link (url).
        """
        # Normalize shapes
        vals = np.array(values)
        if vals.ndim == 2:
                if vals.shape[0] > 1 and class_index < vals.shape[0]:
                        vals = vals[class_index]
                elif vals.shape[1] > 1 and class_index < vals.shape[1]:
                        vals = vals[:, class_index]
                else:
                        vals = vals.reshape(-1)
        elif vals.ndim > 2:
                vals = vals.reshape(-1)

        if tokens is None or not isinstance(tokens, (list, tuple)) or not tokens:
                tokens = text.split()

        # Align lengths
        T = min(len(tokens), len(vals))
        tokens = list(tokens[:T])
        vals = vals[:T]

        # Color scale
        max_abs = float(np.max(np.abs(vals))) if vals.size else 1.0
        max_abs = max(max_abs, 1e-8)

        def color(v: float) -> str:
                a = min(abs(v) / max_abs, 1.0)
                r = int(235 + (255 - 235) * (1 - a)) if v >= 0 else 255
                g = int(245 + (255 - 245) * (1 - a)) if v >= 0 else 235
                b = 255 if v >= 0 else int(235 + (255 - 235) * (1 - a))
                alpha = 0.15 + 0.65 * a
                return f"rgba({r},{g},{b},{alpha:.3f})"

        spans_html: List[str] = []
        for t, v in zip(tokens, vals):
                bg = color(float(v))
                safe_t = (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") or "&nbsp;")
                spans_html.append(f"<span class='tok' style='background:{bg}' title='w={float(v):.4f}'>{safe_t}</span>")

        ctrl = (
                "<div style='display:flex;gap:12px;align-items:center;margin-bottom:10px'>"
                "<label class='mono' style='font-size:12px'>Heatmap: "
                "<select id='modeSel'><option value='signed' selected>signed</option><option value='absolute'>absolute</option></select>"
                "</label>"
                "<label class='mono' style='font-size:12px'>Font size: <input id='fs' type='range' min='10' max='24' step='1' value='14'> <span id='fsval' class='mono'>14</span>px</label>"
                "<button id='csv_all' class='pill'>CSV (all)</button>"
                "<button id='csv' class='pill'>CSV (filtered)</button>"
                "<button id='copycsv' class='pill'>Copy CSV</button>"
                "<label class='mono' style='font-size:12px;display:flex;align-items:center;gap:6px'><input id='showspecial' type='checkbox' checked> show special tokens</label>"
                "</div>"
        )

        body = (
                f"<div class='legend' style='margin-bottom:8px'><span class='pos'>supports class {class_index}</span>"
                f"<span class='neg'>opposes class {class_index}</span></div>"
                + ctrl
                + "<div id='tokview' class='mono' style='line-height:1.9'></div>"
                + "<noscript><div>JavaScript disabled; showing static tokens:</div>" + "".join(spans_html) + "</noscript>"
                + "<h3 style='margin-top:16px'>Top contributing tokens</h3>"
                + "<div style='display:flex;align-items:center;gap:10px;margin:8px 0'>"
                + "<label class='mono' style='font-size:12px'>Sort: <select id='topsort'><option value='absolute'>absolute</option><option value='signed'>signed</option></select></label>"
                + "</div>"
                + "<table id='toptbl' class='tbl'><thead><tr><th>Token</th><th>Weight</th></tr></thead><tbody></tbody></table>"
                + "<h3 style='margin-top:16px'>All tokens (CSV table)</h3>"
                + "<div style='display:flex;align-items:center;gap:10px;margin:8px 0'>"
                + "<label class='mono' style='font-size:12px'>Sort: <select id='csvsort'>"
                + "<option value='abs_desc'>abs desc</option><option value='abs_asc'>abs asc</option>"
                + "<option value='signed_desc'>signed desc</option><option value='signed_asc'>signed asc</option>"
                + "</select></label>"
                + "<label class='mono' style='font-size:12px'>Filter: <input id='csvfilter' placeholder='substring...'></label>"
                + "</div>"
                + "<table id='csvtbl' class='tbl'><thead><tr><th>Token</th><th>Weight</th></tr></thead><tbody></tbody></table>"
                + f"<script>\nconst TOKENS = {json.dumps(tokens)};\nconst WEIGHTS = {json.dumps([float(x) for x in vals.tolist()])};\nconst TOP_K = {int(top_k)};\n" + r"""
function isSpecial(t){ return /^<.*>$/.test(t) || /^\[.*\]$/.test(t); }
function render(mode){
    const cont = document.getElementById('tokview'); if(!cont) return; cont.innerHTML='';
    const maxAbs = Math.max(1e-8, ...WEIGHTS.map(v=>Math.abs(v)));
    const fs = document.getElementById('fs') ? document.getElementById('fs').value : 14;
    const showSpec = document.getElementById('showspecial') ? document.getElementById('showspecial').checked : true;
    TOKENS.forEach((t,i)=>{
        if(!showSpec && isSpecial(t)) return;
        const v = WEIGHTS[i]||0; const a = Math.min(Math.abs(v)/maxAbs, 1.0);
        let bg = ''; let color='';
        if(mode==='absolute'){ bg = `rgba(100,150,255,${0.15+0.65*a})`; color='#bcd4ff'; }
        else { if(v>=0){ bg = `rgba(235,245,255,${0.15+0.65*a})`; color='#9cc4ff'; } else { bg = `rgba(255,235,235,${0.15+0.65*a})`; color='#ff9c9c'; } }
        const span = document.createElement('span'); span.className='tok'; span.style.background=bg; span.style.color=color; span.style.fontSize=fs+'px'; span.title='w='+v.toFixed(4);
        span.textContent = t || '\u00a0'; cont.appendChild(span);
    });
}
function filteredIdx(){
    const showSpec = document.getElementById('showspecial') ? document.getElementById('showspecial').checked : true;
    const q = (document.getElementById('csvfilter')?.value||'').toLowerCase();
    const idx=[]; for(let i=0;i<TOKENS.length;i++){ const t=TOKENS[i]||''; if(!showSpec && isSpecial(t)) continue; if(q && !t.toLowerCase().includes(q)) continue; idx.push(i);} return idx;
}
function renderTop(mode){
    const tb = document.querySelector('#toptbl tbody'); if(!tb) return; tb.innerHTML='';
    let idx = filteredIdx();
    if(mode==='absolute'){ idx.sort((a,b)=>Math.abs(WEIGHTS[b])-Math.abs(WEIGHTS[a])); } else { idx.sort((a,b)=>WEIGHTS[b]-WEIGHTS[a]); }
    idx = idx.slice(0, TOP_K);
    idx.forEach(i=>{ const tr=document.createElement('tr'); const td1=document.createElement('td'); td1.className='mono'; td1.textContent=TOKENS[i]; const td2=document.createElement('td'); td2.className='mono'; td2.style.textAlign='right'; td2.textContent=(WEIGHTS[i]||0).toFixed(4); tr.appendChild(td1); tr.appendChild(td2); tb.appendChild(tr); });
}
function renderCsv(){
    const tb = document.querySelector('#csvtbl tbody'); if(!tb) return; tb.innerHTML='';
    let idx = filteredIdx(); const s=document.getElementById('csvsort')?.value||'abs_desc';
    const cmp={abs_desc:(a,b)=>Math.abs(WEIGHTS[b])-Math.abs(WEIGHTS[a]), abs_asc:(a,b)=>Math.abs(WEIGHTS[a])-Math.abs(WEIGHTS[b]), signed_desc:(a,b)=>WEIGHTS[b]-WEIGHTS[a], signed_asc:(a,b)=>WEIGHTS[a]-WEIGHTS[b]};
    idx.sort(cmp[s]);
    idx.forEach(i=>{ const tr=document.createElement('tr'); const td1=document.createElement('td'); td1.className='mono'; td1.textContent=TOKENS[i]; const td2=document.createElement('td'); td2.className='mono'; td2.style.textAlign='right'; td2.textContent=(WEIGHTS[i]||0).toFixed(6); tr.appendChild(td1); tr.appendChild(td2); tb.appendChild(tr); });
}
const sel=document.getElementById('modeSel'); const fs=document.getElementById('fs'); const fsv=document.getElementById('fsval');
if(sel){ sel.onchange=()=>{render(sel.value); const ts=document.getElementById('topsort'); if(ts) renderTop(ts.value); renderCsv();}; } if(fs){ fs.oninput=()=>{fsv.textContent=fs.value; render(sel?sel.value:'signed');}; }
const ss=document.getElementById('showspecial'); if(ss){ ss.onchange=()=>{ render(sel?sel.value:'signed'); const ts=document.getElementById('topsort'); if(ts) renderTop(ts.value); renderCsv(); }; }
const ts=document.getElementById('topsort'); if(ts){ ts.onchange=()=>renderTop(ts.value); }
const cs=document.getElementById('csvsort'); if(cs){ cs.onchange=()=>renderCsv(); }
const cf=document.getElementById('csvfilter'); if(cf){ cf.oninput=()=>renderCsv(); }
const csv=document.getElementById('csv'); if(csv){ csv.onclick=()=>{
    let rows=['token,weight']; const idx=filteredIdx(); for(const i of idx){ const tok=(TOKENS[i]||'').replaceAll('"','""'); rows.push('"'+tok+'",'+(WEIGHTS[i]||0).toFixed(6)); }
    const blob=new Blob([rows.join('\n')],{type:'text/csv;charset=utf-8;'}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='token_weights_filtered.csv'; a.click();
}; }
const csvAll=document.getElementById('csv_all'); if(csvAll){ csvAll.onclick=()=>{
    let rows=['token,weight']; for(let i=0;i<TOKENS.length;i++){ const t=TOKENS[i]||''; const tok=t.replaceAll('"','""'); rows.push('"'+tok+'",'+(WEIGHTS[i]||0).toFixed(6)); }
    const blob=new Blob([rows.join('\n')],{type:'text/csv;charset=utf-8;'}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='token_weights_all.csv'; a.click();
}; }
const cpy=document.getElementById('copycsv'); if(cpy){ cpy.onclick=async()=>{ let rows=['token,weight']; const idx=filteredIdx(); for(const i of idx){ const tok=(TOKENS[i]||'').replaceAll('"','""'); rows.push('"'+tok+'",'+(WEIGHTS[i]||0).toFixed(6)); } const txt=rows.join('\n'); try{ await navigator.clipboard.writeText(txt);}catch(e){ console.log(e);} }; }
render('signed'); const ts0=document.getElementById('topsort'); renderTop(ts0?ts0.value:'absolute'); renderCsv();
""" + "</script>"
        )
        extra = info or []
        extra = list(extra)
        extra.append(("Chars", str(len(text))))
        extra.append(("Tokens", str(len(tokens))))
        html = _wrap_expl_html(title, url, extra, body, details_extra_html=details_extra_html, header_extra_html=header_extra_html)
        with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)


def _prob_bars_html(probs: List[float], class_names: List[str]) -> str:
    pairs = list(zip(class_names, probs))
    rows = []
    for name, p in pairs:
        p_clamped = max(0.0, min(1.0, float(p)))
        pct = f"{p_clamped*100:.1f}%"
        width = int(p_clamped*100)
        rows.append(
            "<div style='margin:4px 0'>"
            f"<div class='mono' style='font-size:12px;margin-bottom:2px'>{name}: {pct}</div>"
            f"<div style='height:10px;background:#1f2330;border:1px solid #2a2f3a;border-radius:5px;overflow:hidden'><div style='height:100%;width:{width}%;background:linear-gradient(90deg,#2b6cb0,#63b3ed)'></div></div>"
            "</div>"
        )
    return "".join(rows)


def _find_word_spans(text: str, word_tokens: List[str]) -> List[tuple[int, int]]:
    spans: List[tuple[int, int]] = []
    lower = text
    start_search = 0
    for w in word_tokens:
        if not w:
            spans.append((0, 0))
            continue
        idx = lower.find(w, start_search)
        if idx < 0:
            # try from beginning as fallback
            idx = lower.find(w)
        if idx < 0:
            spans.append((0, 0))
        else:
            spans.append((idx, idx + len(w)))
            start_search = idx + len(w)
    return spans


def _align_weights_to_model_tokens(
    text: str,
    word_tokens: List[str],
    word_weights: np.ndarray,
    tokenizer,
    max_length: int,
) -> tuple[List[str], np.ndarray] | None:
    try:
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
    except Exception:
        return None
    offsets = enc.get("offset_mapping")
    input_ids = enc.get("input_ids")
    if offsets is None or input_ids is None:
        return None
    toks = tokenizer.convert_ids_to_tokens(input_ids)
    # Filter special tokens by zero offsets
    piece_tokens: List[str] = []
    piece_offsets: List[tuple[int, int]] = []
    for t, (s, e) in zip(toks, offsets):
        if isinstance(s, (list, tuple)):
            # some tokenizers may nest
            s, e = s  # type: ignore
        if (s, e) == (0, 0) and (t.startswith("<") or t.startswith("[")):
            continue
        piece_tokens.append(t)
        piece_offsets.append((int(s), int(e)))

    # Build model token weights by distributing word weights over overlapping pieces
    model_weights = np.zeros(len(piece_tokens), dtype=float)
    spans = _find_word_spans(text, list(word_tokens))
    for (ws, we), w in zip(spans, word_weights):
        wl = max(1, we - ws)
        for i, (ps, pe) in enumerate(piece_offsets):
            if pe <= ws or ps >= we:
                continue
            overlap = max(0, min(we, pe) - max(ws, ps))
            if overlap > 0:
                model_weights[i] += float(w) * (overlap / wl)
    return piece_tokens, model_weights


def _select_class_values(values: np.ndarray | list, class_index: int = 1) -> np.ndarray:
    vals = np.array(values)
    if vals.ndim == 2:
        # Prefer (T, C)
        if vals.shape[1] > 1 and class_index < vals.shape[1]:
            return vals[:, class_index]
        # Else assume (C, T)
        if vals.shape[0] > 1 and class_index < vals.shape[0]:
            return vals[class_index]
        return vals.reshape(-1)
    elif vals.ndim > 2:
        return vals.reshape(-1)
    return vals


def _read_dom_preds(path: str) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]] | None:
    try:
        if not os.path.exists(path):
            return None
        y: List[int] = []
        p: List[float] = []
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                y.append(int(obj.get("label", 0)))
                p.append(float(obj.get("prob", 0.0)))
                rows.append({"id": obj.get("id"), "label": int(obj.get("label", 0)), "url": obj.get("url")})
        if not y:
            return None
        return np.array(y, dtype=int), np.array(p, dtype=float), rows
    except Exception:
        return None


def _write_dom_preds(path: str, rows: List[Dict[str, Any]], probs: np.ndarray) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            for r, pr in zip(rows, probs.tolist()):
                obj = {"id": r.get("id"), "label": int(r.get("label", 0)), "url": r.get("url"), "prob": float(pr)}
                f.write(json.dumps(obj))
                f.write("\n")
    except Exception:
        pass


def load_preds(model_dir: str, jsonl_path: str, max_length: int = 512, device_preference: str = "cuda", per_device_eval_batch_size: int = 4, cache_path: str | None = None) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    if not os.path.isdir(model_dir):
        raise SystemExit(
            f"[ERROR] model_dir must be a fine-tuned local directory (got '{model_dir}').\n"
            "Pass your trained output directory (e.g., artifacts/markup_run)."
        )
    # Try cache
    if cache_path:
        cached = _read_dom_preds(cache_path)
        if cached is not None:
            return cached
    # Lazy, streaming inference to minimize RAM
    from phisdom.data.schema import iter_jsonl  # local import to avoid cycles
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    processor = MarkupLMProcessor.from_pretrained(model_dir)
    use_cuda = (device_preference == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)  # type: ignore[assignment]
    model.eval()
    if use_cuda:
        try:
            gcount = torch.cuda.device_count()
            gnames = [torch.cuda.get_device_name(i) for i in range(gcount)]
            print(f"[INFO] Report preds using CUDA (gpus={gcount}): {gnames}")
        except Exception:
            print("[INFO] Report preds using CUDA")

    y_list: List[int] = []
    p_list: List[float] = []
    rows_min: List[Dict[str, Any]] = []
    batch_html: List[str] = []
    batch_ids: List[Any] = []
    batch_labels: List[int] = []
    batch_urls: List[str | None] = []
    bs = max(1, int(per_device_eval_batch_size))
    max_bs_cap = 32
    adapt_up = use_cuda
    # For peak mem tracking
    total_mem = None
    if use_cuda:
        try:
            props = torch.cuda.get_device_properties(0)
            total_mem = float(props.total_memory)
        except Exception:
            total_mem = None

    def run_subbatch(start: int, end: int):
        nonlocal y_list, p_list, rows_min
        sb_html = batch_html[start:end]
        sb_ids = batch_ids[start:end]
        sb_labels = batch_labels[start:end]
        sb_urls = batch_urls[start:end]
        enc = processor(html_strings=sb_html, truncation=True, max_length=max_length, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        logits = out.logits.detach().cpu().numpy()
        if logits.ndim == 2 and logits.shape[1] == 2:
            m = np.max(logits, axis=1, keepdims=True)
            e = np.exp(logits - m)
            p1 = (e[:, 1] / (e[:, 0] + e[:, 1]))
        else:
            p1 = 1 / (1 + np.exp(-logits.reshape(-1)))
        y_list.extend(sb_labels)
        p_list.extend(p1.tolist())
        for rid, lab, url in zip(sb_ids, sb_labels, sb_urls):
            rows_min.append({"id": rid, "label": int(lab), "url": url})

    def flush_batch():
        nonlocal batch_html, batch_ids, batch_labels, batch_urls, bs
        if not batch_html:
            return
        with torch.no_grad():
            try:
                if use_cuda:
                    try:
                        torch.cuda.reset_peak_memory_stats()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                # If current batch exceeds bs (e.g., last bump), process in windows of bs
                if len(batch_html) <= bs:
                    run_subbatch(0, len(batch_html))
                else:
                    for k in range(0, len(batch_html), bs):
                        run_subbatch(k, min(len(batch_html), k + bs))
                # Adaptive scale-up: if we used less than ~75% of VRAM, bump bs up
                if adapt_up and total_mem:
                    try:
                        peak = float(torch.cuda.max_memory_allocated())  # type: ignore[attr-defined]
                        if peak > 0 and (total_mem and (peak / total_mem) < 0.75) and bs < max_bs_cap:
                            bs_new = min(max_bs_cap, max(bs + 1, int(bs * 3 / 2)))
                            if bs_new != bs:
                                bs = bs_new
                    except Exception:
                        pass
            except RuntimeError as e:
                # OOM backoff: halve batch and retry in chunks
                if "out of memory" in str(e).lower() and use_cuda:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    new_bs = max(1, bs // 2)
                    if new_bs == bs:
                        # give up and fall back to per-sample
                        new_bs = 1
                    # process in smaller windows
                    for k in range(0, len(batch_html), new_bs):
                        run_subbatch(k, min(len(batch_html), k + new_bs))
                    bs = new_bs
                else:
                    raise
        batch_html = []
        batch_ids = []
        batch_labels = []
        batch_urls = []

    # Progress bar over streaming iterator
    try:
        from tqdm import tqdm  # type: ignore
        it = tqdm(iter_jsonl(jsonl_path), desc=f"predict {os.path.basename(jsonl_path)}", unit="rec")
    except Exception:
        it = iter_jsonl(jsonl_path)

    for r in it:
        html = r.get("html") or ""
        if not html:
            continue
        batch_html.append(html)
        batch_ids.append(r.get("id"))
        try:
            batch_labels.append(int(r.get("label", 0)))
        except Exception:
            batch_labels.append(0)
        batch_urls.append(r.get("url"))
        if len(batch_html) >= bs:
            flush_batch()
    # flush tail
    flush_batch()

    y_true = np.array(y_list, dtype=int)
    probs = np.array(p_list, dtype=float)
    # Save cache if requested
    if cache_path:
        try:
            _write_dom_preds(cache_path, rows_min, probs)
        except Exception:
            pass
    return y_true, probs, rows_min


def plot_curves(y: np.ndarray, p: np.ndarray, out_dir: str, split: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    os.makedirs(out_dir, exist_ok=True)
    # Guard against NaNs in probabilities
    if not np.isfinite(p).all():
        print(f"[WARN] NaN/inf detected in probabilities for split {split}; filtering")
        mask = np.isfinite(p)
        y = y[mask]
        p = p[mask]
    if y.size == 0 or p.size == 0:
        plt.figure(); plt.text(0.5, 0.5, "No valid predictions", ha="center", va="center"); plt.axis("off"); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"pr_curve_{split}.png")); plt.close(); return
    # One-class guard
    if len(set(map(int, y.tolist()))) < 2:
        # Create tiny placeholder images to avoid report breaks
        plt.figure(); plt.text(0.5, 0.5, "One-class split; PR/ROC not computed", ha="center", va="center"); plt.axis("off"); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"pr_curve_{split}.png")); plt.close()
        plt.figure(); plt.text(0.5, 0.5, "One-class split; PR/ROC not computed", ha="center", va="center"); plt.axis("off"); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"roc_curve_{split}.png")); plt.close()
        return

    # PR
    prec, rec, _ = precision_recall_curve(y, p)
    pr_auc = auc(rec, prec)
    plt.figure()
    plt.plot(rec, prec, label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall ({split})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pr_curve_{split}.png"))
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC ({split})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"roc_curve_{split}.png"))
    plt.close()


def plot_reliability(y: np.ndarray, p: np.ndarray, out_dir: str, split: str):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    
    # Filter out NaN/inf values
    mask = np.isfinite(p)
    if not mask.any():
        plt.figure(); plt.text(0.5, 0.5, "All probabilities are NaN/inf", ha="center", va="center"); plt.axis("off"); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"reliability_{split}.png")); plt.close()
        return
    y_clean, p_clean = y[mask], p[mask]
    
    if len(set(map(int, y_clean.tolist()))) < 2:
        plt.figure(); plt.text(0.5, 0.5, "One-class split; reliability not computed", ha="center", va="center"); plt.axis("off"); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"reliability_{split}.png")); plt.close()
        return
    frac_pos, mean_pred = calibration_curve(y_clean, p_clean, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Reliability diagram ({split})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"reliability_{split}.png"))
    plt.close()


def plot_confusion(y: np.ndarray, p: np.ndarray, thr: float, out_dir: str, split: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    os.makedirs(out_dir, exist_ok=True)
    # Handle single-class edge case
    if len(set(map(int, y.tolist()))) < 2:
        plt.figure(); plt.text(0.5, 0.5, "One-class split; confusion not computed", ha="center", va="center"); plt.axis("off"); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"confusion_{split}.png")); plt.close(); return
    # Predictions
    yhat = (p >= thr).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])  # [[TN FP],[FN TP]] with our label order
    tn, fp, fn, tp = cm.ravel()
    # Derived rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tpr
    # Normalized by true-class (row) for heat coloring
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sums, where=row_sums>0)

    plt.figure(figsize=(4.2,4.0))
    im = plt.imshow(norm, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Row %')
    # Annotate with count and row %
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = norm[i, j] * 100 if row_sums[i,0] > 0 else float('nan')
            txt = f"{count}\n{pct:.1f}%"
            plt.text(j, i, txt, ha='center', va='center', fontsize=10, color='#111' if norm[i,j] > 0.6 else '#eee')
    plt.xticks([0,1], ["benign","phish"])
    plt.yticks([0,1], ["benign","phish"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion @thr={thr:.3f} {split}\nTPR={tpr:.3f} FPR={fpr:.3f} Prec={precision:.3f} Rec={recall:.3f}", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"confusion_{split}.png")
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_roc_multi(series: List[Tuple[str, np.ndarray, np.ndarray]], out_dir: str, name: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    for label, y, p in series:
        # Filter out NaN/inf values
        mask = np.isfinite(p)
        if not mask.any():
            print(f"[WARN] All probabilities are NaN/inf for {label}, skipping ROC curve")
            continue
        y_clean, p_clean = y[mask], p[mask]
        
        # Skip if only one class after filtering
        if len(set(map(int, y_clean.tolist()))) < 2:
            print(f"[WARN] Single class after NaN filtering for {label}, skipping ROC curve")
            continue
            
        fpr, tpr, _ = roc_curve(y_clean, p_clean)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (combined)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"roc_{name}_combined.png"))
    plt.close()


def plot_pr_multi(series: List[Tuple[str, np.ndarray, np.ndarray]], out_dir: str, name: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, auc
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    for label, y, p in series:
        # Filter out NaN/inf values
        mask = np.isfinite(p)
        if not mask.any():
            print(f"[WARN] All probabilities are NaN/inf for {label}, skipping PR curve")
            continue
        y_clean, p_clean = y[mask], p[mask]
        
        # Skip if only one class after filtering
        if len(set(map(int, y_clean.tolist()))) < 2:
            print(f"[WARN] Single class after NaN filtering for {label}, skipping PR curve")
            continue
            
        prec, rec, _ = precision_recall_curve(y_clean, p_clean)
        pr_auc = auc(rec, prec)
        plt.plot(rec, prec, label=f"{label} (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (combined)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pr_{name}_combined.png"))
    plt.close()


def plot_reliability_multi(series: List[Tuple[str, np.ndarray, np.ndarray]], out_dir: str, name: str):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    for label, y, p in series:
        # Filter out NaN/inf values
        mask = np.isfinite(p)
        if not mask.any():
            print(f"[WARN] All probabilities are NaN/inf for {label}, skipping reliability")
            continue
        y_clean, p_clean = y[mask], p[mask]
        
        # Skip if only one class after filtering
        if len(set(map(int, y_clean.tolist()))) < 2:
            print(f"[WARN] Single class for {label} after NaN filtering, skipping reliability")
            continue
            
        frac_pos, mean_pred = calibration_curve(y_clean, p_clean, n_bins=10, strategy="uniform")
        plt.plot(mean_pred, frac_pos, marker="o", label=label)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability (combined)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"reliability_{name}_combined.png"))
    plt.close()


def plot_pr_multi_splits(series: List[Tuple[str, np.ndarray, np.ndarray]], out_dir: str, name: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, auc
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    for label, y, p in series:
        # Filter out NaN/inf values first
        mask = np.isfinite(p)
        if not mask.any():
            print(f"[WARN] All probabilities are NaN/inf for {label}, skipping PR curve")
            continue
        y_clean, p_clean = y[mask], p[mask]
        
        # Check for single class after filtering
        if len(set(map(int, y_clean.tolist()))) < 2:
            print(f"[WARN] Single class for {label} after NaN filtering, skipping PR curve")
            continue
            
        prec, rec, _ = precision_recall_curve(y_clean, p_clean)
        pr_auc = auc(rec, prec)
        plt.plot(rec, prec, label=f"{label} (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall (combined {name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pr_{name}_splits_combined.png"))
    plt.close()


def plot_roc_multi_splits(series: List[Tuple[str, np.ndarray, np.ndarray]], out_dir: str, name: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    for label, y, p in series:
        # Filter out NaN/inf values first
        mask = np.isfinite(p)
        if not mask.any():
            print(f"[WARN] All probabilities are NaN/inf for {label}, skipping ROC curve")
            continue
        y_clean, p_clean = y[mask], p[mask]
        
        # Check for single class after filtering
        if len(set(map(int, y_clean.tolist()))) < 2:
            print(f"[WARN] Single class for {label} after NaN filtering, skipping ROC curve")
            continue
            
        fpr, tpr, _ = roc_curve(y_clean, p_clean)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (combined {name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"roc_{name}_splits_combined.png"))
    plt.close()


def plot_reliability_multi_splits(series: List[Tuple[str, np.ndarray, np.ndarray]], out_dir: str, name: str):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    for label, y, p in series:
        # Filter out NaN/inf values first
        mask = np.isfinite(p)
        if not mask.any():
            print(f"[WARN] All probabilities are NaN/inf for {label}, skipping reliability")
            continue
        y_clean, p_clean = y[mask], p[mask]
        
        # Check for single class after filtering
        if len(set(map(int, y_clean.tolist()))) < 2:
            print(f"[WARN] Single class for {label} after NaN filtering, skipping reliability")
            continue
            
        frac_pos, mean_pred = calibration_curve(y_clean, p_clean, n_bins=10, strategy="uniform")
        plt.plot(mean_pred, frac_pos, marker="o", label=label)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Reliability (combined {name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"reliability_{name}_splits_combined.png"))
    plt.close()


def plot_accuracy_curve_multi_splits(series: List[Tuple[str, np.ndarray, np.ndarray]], out_dir: str, name: str):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    thresholds = np.linspace(0, 1, 201)
    plt.figure()
    for label, y, p in series:
        accs = []
        for t in thresholds:
            yhat = (p >= t).astype(int)
            accs.append((yhat == y).mean())
        plt.plot(thresholds, accs, label=label)
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Threshold (combined {name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"accuracy_{name}_splits_combined.png"))
    plt.close()


def plot_train_val_loss(trainer_state_path: str, out_dir: str, name: str):
    import matplotlib.pyplot as plt
    if not os.path.exists(trainer_state_path):
        return
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    logs = state.get("log_history", []) if isinstance(state, dict) else []
    step_loss = [(l.get("step"), l.get("loss")) for l in logs if "loss" in l]
    step_eval = [(l.get("step"), l.get("eval_loss")) for l in logs if "eval_loss" in l]
    if not step_loss and not step_eval:
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    if step_loss:
        x, y = zip(*step_loss)
        plt.plot(x, y, label="train loss")
    if step_eval:
        x, y = zip(*step_eval)
        plt.plot(x, y, label="val loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss ({name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"train_val_loss_{name}.png"))
    plt.close()


def explain_lime(
    model_dir: str,
    samples: List[Dict[str, Any]],
    out_dir: str,
    max_length: int = 512,
    num_features: int = 10,
    num_samples: int = 200,
    max_chars: int = 1500,
    device_preference: str = "cpu",
    tokenizer_mode: str = "whitespace",
):
    try:
        from lime.lime_text import LimeTextExplainer
    except Exception:
        print("[INFO] LIME not installed; skipping LIME explanations")
        return

    processor = MarkupLMProcessor.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    # Use CPU by default for XAI to avoid OOM during many perturbations
    use_cuda = device_preference == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)

    def predict_proba(texts: List[str]) -> np.ndarray:
        # SHAP may pass numpy arrays or scalars; coerce to list[str]
        try:
            import numpy as _np
        except Exception:  # pragma: no cover
            _np = None  # type: ignore
        if isinstance(texts, str):
            texts = [texts]
        elif _np is not None and isinstance(texts, _np.ndarray):  # type: ignore[truthy-bool]
            texts = [
                _ensure_text_str(x, max_chars=max_chars) for x in texts.tolist()
            ]
        elif isinstance(texts, (list, tuple)):
            texts = [
                _ensure_text_str(x, max_chars=max_chars) for x in texts
            ]
        else:
            texts = [_ensure_text_str(texts, max_chars=max_chars)]
        # Replace empty strings with minimal placeholder to avoid zero-token batches
        texts = [t if isinstance(t, str) and t.strip() else "<p>.</p>" for t in texts]
        batch = processor(html_strings=texts, truncation=True, max_length=max_length, return_tensors="pt", padding=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
            logits = out.logits.detach().cpu().numpy()
        # Guard against empty logits
        if logits.size == 0:
            p = np.full((len(texts), 2), 0.5, dtype=float)
            return p
        if logits.ndim == 2 and logits.shape[1] == 2:
            m = logits.max(axis=1, keepdims=True)
            e = np.exp(logits - m)
            p1 = (e[:, 1] / (e[:, 0] + e[:, 1]))
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T
        else:
            p1 = 1 / (1 + np.exp(-logits.reshape(-1)))
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T

    explainer = LimeTextExplainer(class_names=["benign", "phish"])
    os.makedirs(out_dir, exist_ok=True)
    for r in samples:
        html = _ensure_text_str(r.get("html", ""), max_chars=max_chars)
        rid = r.get("id") or "sample"
        url = r.get("url")
        
        # Basic guardrails: skip if text too short or probs are degenerate
        if len(html.split()) < 10:
            print(f"[WARN] LIME skipped {rid}: text too short ({len(html.split())} words)")
            continue
            
        try:
            # Check if prediction is reasonable before explaining
            probs = predict_proba([html])
            if probs is None or len(probs) == 0 or not (0.01 < probs[0][1] < 0.99):
                print(f"[WARN] LIME skipped {rid}: degenerate prediction {probs}")
                continue
                
            exp = explainer.explain_instance(html, predict_proba, num_features=num_features, num_samples=num_samples)
            out_path = os.path.join(out_dir, f"lime_{rid}.html")
            # Extract tokens and weights if available
            try:
                toks, weights = zip(*exp.as_list(label=1))  # class 1: phish
                vals = np.array(weights, dtype=float)
                rtoks = list(toks)
                # Optionally re-align to model tokens
                if tokenizer_mode == "model":
                    tok_ref = getattr(processor, "tokenizer", None)
                    if tok_ref is None:
                        try:
                            tok_ref = AutoTokenizer.from_pretrained(model_dir)
                        except Exception:
                            tok_ref = None
                    aligned = _align_weights_to_model_tokens(html, rtoks, vals, tok_ref, max_length) if tok_ref is not None else None
                    if aligned is not None:
                        rtoks, vals = aligned
                # Use our renderer for a consistent look
                # Estimate probability for the shown input
                pmat = predict_proba([html])
                probs = pmat[0].tolist()
                bars = _prob_bars_html(probs, ["benign","phish"]) 
                pred_label = "phish" if probs[1] >= probs[0] else "benign"
                pred_prob = max(probs)
                header_pill = f"<div class='pill' style='margin-left:8px;background:#1a2a1a;border-color:#245c2a;color:#a7e3a7'>pred: {pred_label} · {pred_prob:.2%}</div>"
                meta = [("ID", str(rid)), ("Label", str(int(r.get("label", 0)))), ("Explainer", "LIME (DOM)"), ("Device", str(device)), ("P(benign)", f"{probs[0]:.3f}"), ("P(phish)", f"{probs[1]:.3f}")]
                _save_shap_text_html(out_path, html, rtoks, vals, class_index=1, title="LIME (DOM)", url=url, info=meta, details_extra_html=f"<div style='margin-top:8px'>{bars}</div>", header_extra_html=header_pill)
            except Exception:
                exp.save_to_file(out_path)
                _inject_link_into_html(out_path, url)
            print(f"[INFO] Wrote LIME: {out_path}")
        except Exception as e:
            print(f"[WARN] LIME failed on {rid}: {e}")


def explain_shap(
    model_dir: str,
    samples: List[Dict[str, Any]],
    out_dir: str,
    max_length: int = 512,
    num_samples: int = 100,
    max_chars: int = 1500,
    background_size: int = 3,
    device_preference: str = "cpu",
    tokenizer_mode: str = "whitespace",
):
    """Efficient SHAP for text using Text masker and batched prediction."""
    try:
        import shap
    except Exception:
        print("[INFO] SHAP not installed; skipping SHAP explanations")
        return

    processor = MarkupLMProcessor.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    use_cuda = device_preference == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)

    def predict_proba(texts: List[str]) -> np.ndarray:
        # SHAP may pass numpy arrays or scalars; coerce to list[str]
        try:
            import numpy as _np
        except Exception:  # pragma: no cover
            _np = None  # type: ignore
        if isinstance(texts, str):
            texts = [texts]
        elif _np is not None and isinstance(texts, _np.ndarray):  # type: ignore[truthy-bool]
            texts = [
                _ensure_text_str(x, max_chars=max_chars) for x in texts.tolist()
            ]
        elif isinstance(texts, (list, tuple)):
            texts = [
                _ensure_text_str(x, max_chars=max_chars) for x in texts
            ]
        else:
            texts = [_ensure_text_str(texts, max_chars=max_chars)]
        batch = processor(html_strings=texts, truncation=True, max_length=max_length, return_tensors="pt", padding=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
            logits = out.logits.detach().cpu().numpy()
        if logits.ndim == 2 and logits.shape[1] == 2:
            m = logits.max(axis=1, keepdims=True)
            e = np.exp(logits - m)
            p1 = (e[:, 1] / (e[:, 0] + e[:, 1]))
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T
        else:
            p1 = 1 / (1 + np.exp(-logits.reshape(-1)))
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T

    texts = [_ensure_text_str(r.get("html", ""), max_chars=max_chars) for r in samples]
    ids = [r.get("id") or f"sample_{i}" for i, r in enumerate(samples)]
    urls = [r.get("url") for r in samples]
    # Filter out empty/untokenizable strings early to avoid SHAP internal errors
    filtered = [(t, i, u) for t, i, u in zip(texts, ids, urls) if isinstance(t, str) and t.strip() and len(t.split()) > 1]
    if not filtered:
        print("[WARN] No suitable DOM texts for SHAP after filtering; skipping")
        return
    if not texts:
        return
    os.makedirs(out_dir, exist_ok=True)
    explainer = None
    try:
        # Prefer the dedicated Text explainer if available (more robust)
        Expl = getattr(__import__("shap.explainers", fromlist=["Text"]), "Text")  # type: ignore[attr-defined]
        explainer = Expl(predict_proba)
    except Exception:
        try:
            masker = shap.maskers.Text()  # type: ignore[attr-defined]
            explainer = shap.Explainer(predict_proba, masker)
        except Exception as e:
            print(f"[WARN] SHAP Text Explainer init failed for DOM: {e}; skipping DOM SHAP")
            return

    # Compute explanations per-sample using raw string input
    sel_n = max(1, num_samples // 10)
    for text, rid, url in filtered[:sel_n]:
        out_path = os.path.join(out_dir, f"shap_{rid}.html")
        
        # Additional guardrails for SHAP
        if len(text.split()) < 15:
            print(f"[WARN] SHAP skipped {rid}: text too short for reliable explanation")
            continue
            
        try:
            # Check prediction quality before explaining
            test_probs = predict_proba([text])
            if test_probs is None or len(test_probs) == 0 or not (0.01 < test_probs[0][1] < 0.99):
                print(f"[WARN] SHAP skipped {rid}: degenerate prediction")
                continue
                
            text = _ensure_text_str(text, max_chars=max_chars)
            exp_all = explainer([text], max_evals=num_samples)  # type: ignore[call-arg]
            # Get the first explanation
            try:
                exp = exp_all[0]
            except Exception:
                exp = exp_all
            try:
                import shap
                html_obj = shap.plots.text(exp, display=False)
                shap.save_html(out_path, html_obj)
                # Also add prediction pill to the top
                pmat = predict_proba([text])
                probs = pmat[0].tolist()
                pred_label = "phish" if probs[1] >= probs[0] else "benign"
                pred_prob = max(probs)
                header_pill = f"<span class='pill' style='background:#1a2a1a;border:1px solid #245c2a;color:#a7e3a7'>pred: {pred_label} · {pred_prob:.2%}</span>"
                _inject_link_into_html(out_path, url, extra_html=header_pill)
            except Exception:
                # Fallback: custom token-level HTML renderer
                vals = _select_class_values(getattr(exp, "values", []), class_index=1)
                toks = list(getattr(exp, "data", [])) if hasattr(exp, "data") else None
                if isinstance(toks, str):
                    toks = toks.split()
                # Align to model tokens if requested
                if tokenizer_mode == "model" and toks is not None:
                    tok_ref = getattr(processor, "tokenizer", None)
                    if tok_ref is None:
                        try:
                            tok_ref = AutoTokenizer.from_pretrained(model_dir)
                        except Exception:
                            tok_ref = None
                    aligned = _align_weights_to_model_tokens(text, list(toks), vals, tok_ref, max_length) if tok_ref is not None else None
                    if aligned is not None:
                        toks, vals = aligned
                pmat = predict_proba([text])
                probs = pmat[0].tolist()
                bars = _prob_bars_html(probs, ["benign","phish"]) 
                pred_label = "phish" if probs[1] >= probs[0] else "benign"
                pred_prob = max(probs)
                header_pill = f"<div class='pill' style='margin-left:8px;background:#1a2a1a;border-color:#245c2a;color:#a7e3a7'>pred: {pred_label} · {pred_prob:.2%}</div>"
                meta = [("ID", str(rid)), ("Explainer", "SHAP (DOM)"), ("Device", str(device)), ("P(benign)", f"{probs[0]:.3f}"), ("P(phish)", f"{probs[1]:.3f}")]
                _save_shap_text_html(out_path, text, toks, vals, class_index=1, title="SHAP (DOM)", url=url, info=meta, details_extra_html=f"<div style='margin-top:8px'>{bars}</div>", header_extra_html=header_pill)
            _inject_link_into_html(out_path, url)
            print(f"[INFO] Wrote SHAP: {out_path}")
        except Exception as e:
            print(f"[WARN] SHAP failed on {rid}: {e}")


def explain_lime_js(
    js_dir: str,
    samples: List[Dict[str, Any]],
    out_dir: str,
    max_length: int = 512,
    num_features: int = 10,
    num_samples: int = 200,
    max_chars: int = 1500,
    device_preference: str = "cpu",
    tokenizer_mode: str = "whitespace",
):
    try:
        from lime.lime_text import LimeTextExplainer
    except Exception:
        print("[INFO] LIME not installed; skipping JS LIME explanations")
        return
    try:
        tok = AutoTokenizer.from_pretrained(js_dir)
        enc = T5EncoderModel.from_pretrained(js_dir)
    except Exception as e:
        print(f"[INFO] Cannot load JS model: {e}")
        return
    use_cuda = device_preference == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    enc = enc.to(device)  # type: ignore[assignment]
    enc.eval()
    # Load classifier head
    clf_path = os.path.join(js_dir, "classifier.pt")
    if not os.path.exists(clf_path):
        print("[INFO] JS classifier.pt not found; skipping JS LIME")
        return
    st = torch.load(clf_path, map_location=device)
    W = st.get("weight")
    B = st.get("bias")
    if W is None:
        print("[INFO] JS classifier weights missing; skipping JS LIME")
        return
    W = W.to(device)
    B = B.to(device) if B is not None else None

    def predict_proba(texts: List[str]) -> np.ndarray:
        # SHAP may pass numpy arrays or scalars; coerce to list[str]
        try:
            import numpy as _np
        except Exception:  # pragma: no cover
            _np = None  # type: ignore
        if isinstance(texts, str):
            texts = [texts]
        elif _np is not None and isinstance(texts, _np.ndarray):  # type: ignore[truthy-bool]
            texts = [
                _ensure_text_str(x, max_chars=max_chars) for x in texts.tolist()
            ]
        elif isinstance(texts, (list, tuple)):
            texts = [
                _ensure_text_str(x, max_chars=max_chars) for x in texts
            ]
        else:
            texts = [_ensure_text_str(texts, max_chars=max_chars)]
        # Replace empty strings with minimal placeholder
        texts = [t if isinstance(t, str) and t.strip() else "." for t in texts]
        batch = tok(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = enc(**batch)
            last_hidden = out.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).type_as(last_hidden)
            pooled = (last_hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
            logits = pooled @ W.T
            if B is not None:
                logits = logits + B
            logits = logits.detach().cpu().numpy()
        if logits.size == 0:
            return np.full((len(texts), 2), 0.5, dtype=float)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        p1 = (e[:, 1] / (e[:, 0] + e[:, 1]))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    # Build LIME explanations
    try:
        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(class_names=["benign", "phish"])
    except Exception:
        print("[INFO] LIME not installed; skipping JS LIME explanations")
        return
    os.makedirs(out_dir, exist_ok=True)
    for r in samples:
        text = _ensure_text_str(r.get("text", ""), max_chars=max_chars)
        rid = r.get("id") or "sample"
        url = r.get("url")
        
        # Basic guardrails: skip if text too short or probs are degenerate
        if len(text.split()) < 10:
            print(f"[WARN] JS LIME skipped {rid}: text too short ({len(text.split())} words)")
            continue
            
        try:
            # Check if prediction is reasonable before explaining
            probs = predict_proba([text])
            if probs is None or len(probs) == 0 or not (0.01 < probs[0][1] < 0.99):
                print(f"[WARN] JS LIME skipped {rid}: degenerate prediction")
                continue
                
            exp = explainer.explain_instance(text, predict_proba, num_features=num_features, num_samples=num_samples)
            out_path = os.path.join(out_dir, f"lime_js_{rid}.html")
            try:
                toks, weights = zip(*exp.as_list(label=1))
                vals = np.array(weights, dtype=float)
                rtoks = list(toks)
                if tokenizer_mode == "model":
                    aligned = _align_weights_to_model_tokens(text, rtoks, vals, tok, max_length)
                    if aligned is not None:
                        rtoks, vals = aligned
                pmat = predict_proba([text])
                probs = pmat[0].tolist()
                bars = _prob_bars_html(probs, ["benign","phish"]) 
                pred_label = "phish" if probs[1] >= probs[0] else "benign"
                pred_prob = max(probs)
                header_pill = f"<div class='pill' style='margin-left:8px;background:#1a2a1a;border-color:#245c2a;color:#a7e3a7'>pred: {pred_label} · {pred_prob:.2%}</div>"
                meta = [("ID", str(rid)), ("Label", str(int(r.get("label", 0)))), ("Explainer", "LIME (JS)"), ("Device", str(device)), ("P(benign)", f"{probs[0]:.3f}"), ("P(phish)", f"{probs[1]:.3f}")]
                _save_shap_text_html(out_path, text, rtoks, vals, class_index=1, title="LIME (JS)", url=url, info=meta, details_extra_html=f"<div style='margin-top:8px'>{bars}</div>", header_extra_html=header_pill)
            except Exception:
                exp.save_to_file(out_path)
                _inject_link_into_html(out_path, url)
            print(f"[INFO] Wrote JS LIME: {out_path}")
        except Exception as e:
            print(f"[WARN] JS LIME failed on {rid}: {e}")


def explain_shap_js(
    js_dir: str,
    samples: List[Dict[str, Any]],
    out_dir: str,
    max_length: int = 512,
    num_samples: int = 100,
    max_chars: int = 1500,
    background_size: int = 3,
    device_preference: str = "cpu",
    tokenizer_mode: str = "whitespace",
):
    """Efficient SHAP for JS text using Text masker and our encoder head."""
    try:
        import shap
    except Exception:
        print("[INFO] SHAP not installed; skipping JS SHAP explanations")
        return
    try:
        tok = AutoTokenizer.from_pretrained(js_dir)
        enc = T5EncoderModel.from_pretrained(js_dir)
    except Exception as e:
        print(f"[INFO] Cannot load JS model: {e}")
        return
    use_cuda = device_preference == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    enc = enc.to(device)  # type: ignore[assignment]
    enc.eval()
    # Load classifier head
    clf_path = os.path.join(js_dir, "classifier.pt")
    if not os.path.exists(clf_path):
        print("[INFO] JS classifier.pt not found; skipping JS SHAP")
        return
    st = torch.load(clf_path, map_location=device)
    W = st.get("weight")
    B = st.get("bias")
    if W is None:
        print("[INFO] JS classifier weights missing; skipping JS SHAP")
        return
    W = W.to(device)
    B = B.to(device) if B is not None else None

    def predict_proba(texts: List[str]) -> np.ndarray:
        # SHAP may pass numpy arrays or scalars; coerce to list[str]
        try:
            import numpy as _np
        except Exception:  # pragma: no cover
            _np = None  # type: ignore
        if isinstance(texts, str):
            texts = [texts]
        elif _np is not None and isinstance(texts, _np.ndarray):  # type: ignore[truthy-bool]
            texts = [
                _ensure_text_str(x, max_chars=max_chars) for x in texts.tolist()
            ]
        elif isinstance(texts, (list, tuple)):
            texts = [
                _ensure_text_str(x, max_chars=max_chars) for x in texts
            ]
        else:
            texts = [_ensure_text_str(texts, max_chars=max_chars)]
        batch = tok(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = enc(**batch)
            last_hidden = out.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).type_as(last_hidden)
            pooled = (last_hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
            logits = pooled @ W.T
            if B is not None:
                logits = logits + B
            logits = logits.detach().cpu().numpy()
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        p1 = (e[:, 1] / (e[:, 0] + e[:, 1]))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    texts = [_ensure_text_str(r.get("text", ""), max_chars=max_chars) for r in samples if r.get("text")]
    ids = [r.get("id") or f"sample_{i}" for i, r in enumerate(samples) if r.get("text")]
    urls = [r.get("url") for r in samples if r.get("text")]
    # Filter out empty/untokenizable strings
    filtered = [(t, i, u) for t, i, u in zip(texts, ids, urls) if isinstance(t, str) and t.strip() and len(t.split()) > 1]
    if not filtered:
        print("[WARN] No suitable JS texts for SHAP after filtering; skipping")
        return
    os.makedirs(out_dir, exist_ok=True)
    explainer = None
    try:
        Expl = getattr(__import__("shap.explainers", fromlist=["Text"]), "Text")  # type: ignore[attr-defined]
        explainer = Expl(predict_proba)
    except Exception:
        try:
            masker = shap.maskers.Text()  # type: ignore[attr-defined]
            explainer = shap.Explainer(predict_proba, masker)
        except Exception as e:
            print(f"[WARN] SHAP Text Explainer init failed for JS: {e}; skipping JS SHAP")
            return
    sel_n = max(1, num_samples // 10)
    for text, rid, url in filtered[:sel_n]:
        out_path = os.path.join(out_dir, f"shap_js_{rid}.html")
        
        # Additional guardrails for JS SHAP
        if len(text.split()) < 15:
            print(f"[WARN] JS SHAP skipped {rid}: text too short for reliable explanation")
            continue
            
        try:
            # Check prediction quality before explaining
            test_probs = predict_proba([text])
            if test_probs is None or len(test_probs) == 0 or not (0.01 < test_probs[0][1] < 0.99):
                print(f"[WARN] JS SHAP skipped {rid}: degenerate prediction")
                continue
                
            text = _ensure_text_str(text, max_chars=max_chars)
            exp_all = explainer([text], max_evals=num_samples)  # type: ignore[call-arg]
            try:
                exp = exp_all[0]
            except Exception:
                exp = exp_all
            try:
                html_obj = shap.plots.text(exp, display=False)
                shap.save_html(out_path, html_obj)
                pmat = predict_proba([text])
                probs = pmat[0].tolist()
                pred_label = "phish" if probs[1] >= probs[0] else "benign"
                pred_prob = max(probs)
                header_pill = f"<span class='pill' style='background:#1a2a1a;border:1px solid #245c2a;color:#a7e3a7'>pred: {pred_label} · {pred_prob:.2%}</span>"
                _inject_link_into_html(out_path, url, extra_html=header_pill)
            except Exception:
                vals = _select_class_values(getattr(exp, "values", []), class_index=1)
                toks = list(getattr(exp, "data", [])) if hasattr(exp, "data") else None
                if isinstance(toks, str):
                    toks = toks.split()
                if tokenizer_mode == "model" and toks is not None:
                    aligned = _align_weights_to_model_tokens(text, list(toks), vals, tok, max_length)
                    if aligned is not None:
                        toks, vals = aligned
                pmat = predict_proba([text])
                probs = pmat[0].tolist()
                bars = _prob_bars_html(probs, ["benign","phish"]) 
                pred_label = "phish" if probs[1] >= probs[0] else "benign"
                pred_prob = max(probs)
                header_pill = f"<div class='pill' style='margin-left:8px;background:#1a2a1a;border-color:#245c2a;color:#a7e3a7'>pred: {pred_label} · {pred_prob:.2%}</div>"
                meta = [("ID", str(rid)), ("Explainer", "SHAP (JS)"), ("Device", str(device)), ("P(benign)", f"{probs[0]:.3f}"), ("P(phish)", f"{probs[1]:.3f}")]
                _save_shap_text_html(out_path, text, toks, vals, class_index=1, title="SHAP (JS)", url=url, info=meta, details_extra_html=f"<div style='margin-top:8px'>{bars}</div>", header_extra_html=header_pill)
            _inject_link_into_html(out_path, url)
            print(f"[INFO] Wrote JS SHAP: {out_path}")
        except Exception as e:
            print(f"[WARN] JS SHAP failed on {rid}: {e}")


def main():
    # Keep logs clean from expected HF warnings (e.g., head init)
    hf_logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description="Generate evaluation plots and explanations (extended version)")
    parser.add_argument("--model-dir", default="artifacts/markup_run")
    parser.add_argument("--out-dir", default=None, help="Optional explicit output directory (overrides <model-dir>/report)")
    parser.add_argument("--js-dir", default="artifacts/js_codet5p")
    parser.add_argument("--fusion-dir", default="artifacts/fusion")
    parser.add_argument("--xfusion-dir", default=None, help="Optional explicit cross-attention fusion directory (artifacts/fusion_xattn by default)")
    parser.add_argument("--xfusion-diag", default=None, help="Optional explicit XFusion diagnostics JSON path")
    parser.add_argument("--meta-fusion-dir", default="artifacts/fusion_meta", help="Directory containing meta-fusion (all heads) predictions")
    parser.add_argument("--heads-dirs", nargs="*", default=None, help="Optional list of additional head directories to include (expects calibration_eval.json inside each)")
    parser.add_argument("--cascade-dir", default=None, help="Optional cascade directory (overrides artifacts/cascade)")
    parser.add_argument("--train-jsonl", default="data/pages_train.jsonl")
    parser.add_argument("--val-jsonl", default="data/pages_val.jsonl")
    parser.add_argument("--test-jsonl", default="data/pages_test.jsonl")
    parser.add_argument("--full-jsonl", default="data/pages.jsonl", help="Path to full (latest) dataset for drift/count checks")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lime", action="store_true", help="Generate LIME explanations (slow)")
    parser.add_argument("--shap", action="store_true", help="Generate SHAP explanations (slow)")
    parser.add_argument("--num-expl", type=int, default=2, help="Number of samples per class to explain")
    parser.add_argument("--xai-device", choices=["cpu", "cuda"], default="cpu", help="Device for LIME/SHAP inference (default cpu)")
    parser.add_argument("--xai-max-chars", type=int, default=1500, help="Max characters of text/HTML to use per sample for XAI")
    parser.add_argument("--xai-num-samples", type=int, default=200, help="Perturbation samples for LIME/SHAP (smaller is faster)")
    parser.add_argument("--xai-background", type=int, default=3, help="Background size for SHAP KernelExplainer")
    parser.add_argument("--xai-tokenizer", choices=["whitespace", "model"], default="whitespace", help="Tokenization used for rendering explanations")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device for prediction during reporting (DOM/JS)")
    parser.add_argument("--eval-batch", type=int, default=4, help="Per-device eval batch size for reporting")
    parser.add_argument("--splits-version", default=None, help="Optional splits version tag or path to v2 JSON for provenance")
    args = parser.parse_args()

    ensure_deps()

    # Resolve report output directory (allow external specification)
    report_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(args.model_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    # Load calibration thresholds if present
    thr_95 = thr_90 = 0.5
    cal_path = os.path.join(args.model_dir, "calibration.json")
    if os.path.exists(cal_path):
        try:
            with open(cal_path, "r", encoding="utf-8") as f:
                cal = json.load(f)
            thr_95 = float(cal.get("thresholds", {}).get("0.95", {}).get("threshold", 0.5))
            thr_90 = float(
                cal.get("thresholds", {}).get("0.90", {}).get("threshold",
                cal.get("thresholds", {}).get("0.9", {}).get("threshold", 0.5))
            )
        except Exception:
            pass

    # Predictions (DOM) with caching to avoid recomputing
    cache_dir = os.path.join(args.model_dir, "report", "preds_cache")
    os.makedirs(cache_dir, exist_ok=True)
    y_tr, p_tr, rows_tr = load_preds(
        args.model_dir,
        args.train_jsonl,
        args.max_length,
        args.device,
        args.eval_batch,
        cache_path=os.path.join(cache_dir, "train_dom.jsonl"),
    )
    # Prefer existing preds_{split}.jsonl in model_dir for val/test if present
    val_cache = os.path.join(args.model_dir, "preds_val.jsonl")
    test_cache = os.path.join(args.model_dir, "preds_test.jsonl")
    y_va, p_va, rows_va = load_preds(
        args.model_dir,
        args.val_jsonl,
        args.max_length,
        args.device,
        args.eval_batch,
        cache_path=val_cache if os.path.exists(val_cache) else os.path.join(cache_dir, "val_dom.jsonl"),
    )
    y_te, p_te, rows_te = load_preds(
        args.model_dir,
        args.test_jsonl,
        args.max_length,
        args.device,
        args.eval_batch,
        cache_path=test_cache if os.path.exists(test_cache) else os.path.join(cache_dir, "test_dom.jsonl"),
    )

    # Auto-switch to *_full split files if a split is single-class or effectively degenerate
    # (very tiny minority class) and the full variant exists.
    def _is_degenerate_split(y_arr: np.ndarray, min_minor: int = 5) -> bool:
        try:
            vals = np.asarray(y_arr, dtype=int)
            n_pos = int((vals == 1).sum())
            n_neg = int((vals == 0).sum())
            if (n_pos == 0) or (n_neg == 0):
                return True
            return min(n_pos, n_neg) < min_minor
        except Exception:
            try:
                classes = set(map(int, y_arr.tolist()))
                return len(classes) < 2
            except Exception:
                return False

    def _with_full_suffix(path: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}_full{ext}"

    used_splits: Dict[str, str] = {"train": os.path.abspath(args.train_jsonl), "val": os.path.abspath(args.val_jsonl), "test": os.path.abspath(args.test_jsonl)}

    if _is_degenerate_split(y_va):
        val_full = _with_full_suffix(args.val_jsonl)
        if os.path.exists(val_full):
            print(f"[report][INFO] Validation split is one-class; switching to {val_full}")
            y_va, p_va, rows_va = load_preds(
                args.model_dir,
                val_full,
                args.max_length,
                args.device,
                args.eval_batch,
                cache_path=val_cache if os.path.exists(val_cache) else os.path.join(cache_dir, "val_dom.jsonl"),
            )
        used_splits["val"] = os.path.abspath(val_full)

    if _is_degenerate_split(y_te):
        test_full = _with_full_suffix(args.test_jsonl)
        if os.path.exists(test_full):
            print(f"[report][INFO] Test split is one-class; switching to {test_full}")
            y_te, p_te, rows_te = load_preds(
                args.model_dir,
                test_full,
                args.max_length,
                args.device,
                args.eval_batch,
                cache_path=test_cache if os.path.exists(test_cache) else os.path.join(cache_dir, "test_dom.jsonl"),
            )
        used_splits["test"] = os.path.abspath(test_full)

    # Load JS preds if available (compute if missing)
    def read_preds_arrays(path: str):
        if not os.path.exists(path):
            return None
        y: List[int] = []
        p: List[float] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                y.append(int(obj.get("label", 0)))
                p.append(float(obj.get("prob", 0.0)))
        return np.array(y, dtype=int), np.array(p, dtype=float)

    def compute_js_preds(jsonl_path: str):
        try:
            tok = AutoTokenizer.from_pretrained(args.js_dir)
            enc = T5EncoderModel.from_pretrained(args.js_dir)
        except Exception:
            return None
        # Load dataset
        from phisdom.data.js import JsonlJsDataset
        ds = JsonlJsDataset(jsonl_path)
        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
        enc = enc.to(device)  # type: ignore[assignment]
        enc.eval()
        # Load classifier head
        clf_path = os.path.join(args.js_dir, "classifier.pt")
        W = B = None
        if os.path.exists(clf_path):
            st = torch.load(clf_path, map_location=device)
            W = st.get("weight")
            B = st.get("bias")
            if W is not None:
                W = W.to(device)
                if B is not None:
                    B = B.to(device)
        if W is None:
            return None
        ids: List[Any] = []
        probs: List[float] = []
        bs = 8
        max_bs_cap = 64
        total_mem = None
        if device.type == "cuda":
            try:
                total_mem = float(torch.cuda.get_device_properties(0).total_memory)
            except Exception:
                total_mem = None
        with torch.no_grad():
            i = 0
            while i < len(ds):
                end = min(len(ds), i + bs)
                # Gather batch rows via index to support lazy datasets
                batch_rows = [ds[j] for j in range(i, end)]
                texts = [r.get("text", "") for r in batch_rows]
                try:
                    if device.type == "cuda":
                        try:
                            torch.cuda.reset_peak_memory_stats()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    batch_tok = tok(texts, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
                    batch_tok = {k: v.to(device) for k, v in batch_tok.items()}
                    out = enc(**batch_tok)
                    last_hidden = out.last_hidden_state
                    mask = batch_tok["attention_mask"].unsqueeze(-1).type_as(last_hidden)
                    pooled = (last_hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
                    logits = pooled @ W.T
                    if B is not None:
                        logits = logits + B
                    logits = logits.detach().cpu().numpy()
                    m = np.max(logits, axis=1, keepdims=True)
                    e = np.exp(logits - m)
                    p1 = (e[:, 1] / (e[:, 0] + e[:, 1]))
                    # success path - only add ids/probs after successful processing
                    ids.extend([r.get("id") for r in batch_rows])
                    probs.extend(p1.tolist())
                    # adaptive increase if headroom
                    if device.type == "cuda" and total_mem:
                        try:
                            peak = float(torch.cuda.max_memory_allocated())  # type: ignore[attr-defined]
                            if peak > 0 and peak / total_mem < 0.60 and bs < max_bs_cap:
                                bs = min(max_bs_cap, max(bs + 1, int(bs * 3 / 2)))
                        except Exception:
                            pass
                    i = end
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and device.type == "cuda" and bs > 1:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        # back off and retry smaller batch (no need to clean up ids since we never added them)
                        bs = max(1, bs // 2)
                        continue
                    else:
                        raise
        y = np.array([int(ds[k].get("label", 0)) for k in range(len(ds))], dtype=int)
        return ids, y, np.array(probs, dtype=float)

    # Prefer extended (_full) predictions if they exist to improve coverage and avoid degenerate splits
    def prefer_full(dir_path: str, base: str) -> str:
        root, ext = os.path.splitext(base)
        full = f"{root}_full{ext}"
        p_full = os.path.join(dir_path, full)
        p_base = os.path.join(dir_path, base)
        return p_full if os.path.exists(p_full) else p_base

    js_val_path = prefer_full(args.js_dir, "preds_val.jsonl")
    js_test_path = prefer_full(args.js_dir, "preds_test.jsonl")
    fu_val_path = prefer_full(args.fusion_dir, "preds_val.jsonl")
    fu_test_path = prefer_full(args.fusion_dir, "preds_test.jsonl")
    # Meta-fusion (all heads) optional
    meta_val_path = prefer_full(args.meta_fusion_dir, "preds_val.jsonl")
    meta_test_path = prefer_full(args.meta_fusion_dir, "preds_test.jsonl")
    # Optional: cross-attention fusion
    # Allow overriding xfusion dir via CLI
    xfu_dir = args.xfusion_dir if args.xfusion_dir else (os.path.join(os.path.dirname(args.fusion_dir), "fusion_xattn") if os.path.isabs(args.fusion_dir) else os.path.join("artifacts", "fusion_xattn"))
    xfu_val_path = prefer_full(xfu_dir, "preds_val.jsonl")
    xfu_test_path = prefer_full(xfu_dir, "preds_test.jsonl")

    js_val = read_preds_arrays(js_val_path)
    js_test = read_preds_arrays(js_test_path)
    # Compute JS train preds if possible
    js_train = compute_js_preds(args.train_jsonl)
    fu_val = read_preds_arrays(fu_val_path)
    fu_test = read_preds_arrays(fu_test_path)
    xfu_val = read_preds_arrays(xfu_val_path)
    xfu_test = read_preds_arrays(xfu_test_path)
    meta_val = read_preds_arrays(meta_val_path)
    meta_test = read_preds_arrays(meta_test_path)

    # Curves & reliability
    plot_curves(y_tr, p_tr, report_dir, "train")
    plot_curves(y_va, p_va, report_dir, "val")
    plot_curves(y_te, p_te, report_dir, "test")
    plot_reliability(y_va, p_va, report_dir, "val")
    plot_reliability(y_te, p_te, report_dir, "test")

    # JS curves if present
    if js_val and js_test:
        y_va_js, p_va_js = js_val
        y_te_js, p_te_js = js_test
        plot_curves(y_va_js, p_va_js, report_dir, "val_js")
        plot_curves(y_te_js, p_te_js, report_dir, "test_js")
        plot_reliability(y_va_js, p_va_js, report_dir, "val_js")
        plot_reliability(y_te_js, p_te_js, report_dir, "test_js")

    # Fusion curves if present
    if fu_val and fu_test:
        y_va_fu, p_va_fu = fu_val
        y_te_fu, p_te_fu = fu_test
        plot_curves(y_va_fu, p_va_fu, report_dir, "val_fused")
        plot_curves(y_te_fu, p_te_fu, report_dir, "test_fused")
        plot_reliability(y_va_fu, p_va_fu, report_dir, "val_fused")
        plot_reliability(y_te_fu, p_te_fu, report_dir, "test_fused")
    # XAttn Fusion curves if present
    if xfu_val and xfu_test:
        y_va_xf, p_va_xf = xfu_val
        y_te_xf, p_te_xf = xfu_test
        plot_curves(y_va_xf, p_va_xf, report_dir, "val_xfusion")
        plot_curves(y_te_xf, p_te_xf, report_dir, "test_xfusion")
        plot_reliability(y_va_xf, p_va_xf, report_dir, "val_xfusion")
        plot_reliability(y_te_xf, p_te_xf, report_dir, "test_xfusion")
    # Meta-fusion curves if present
    if meta_val and meta_test:
        y_va_meta, p_va_meta = meta_val
        y_te_meta, p_te_meta = meta_test
        plot_curves(y_va_meta, p_va_meta, report_dir, "val_meta")
        plot_curves(y_te_meta, p_te_meta, report_dir, "test_meta")
        plot_reliability(y_va_meta, p_va_meta, report_dir, "val_meta")
        plot_reliability(y_te_meta, p_te_meta, report_dir, "test_meta")

    # Build multi-split overlays for DOM
    plot_pr_multi_splits([("train", y_tr, p_tr), ("val", y_va, p_va), ("test", y_te, p_te)], report_dir, "dom")
    plot_roc_multi_splits([("train", y_tr, p_tr), ("val", y_va, p_va), ("test", y_te, p_te)], report_dir, "dom")
    plot_reliability_multi_splits([("train", y_tr, p_tr), ("val", y_va, p_va), ("test", y_te, p_te)], report_dir, "dom")
    plot_accuracy_curve_multi_splits([("train", y_tr, p_tr), ("val", y_va, p_va), ("test", y_te, p_te)], report_dir, "dom")

    # Multi-split overlays for JS if available
    if js_train and js_val and js_test:
        ids_tr_js, y_tr_js, p_tr_js = js_train
        y_va_js, p_va_js = js_val
        y_te_js, p_te_js = js_test
        plot_pr_multi_splits([("train", y_tr_js, p_tr_js), ("val", y_va_js, p_va_js), ("test", y_te_js, p_te_js)], report_dir, "js")
        plot_roc_multi_splits([("train", y_tr_js, p_tr_js), ("val", y_va_js, p_va_js), ("test", y_te_js, p_te_js)], report_dir, "js")
        plot_reliability_multi_splits([("train", y_tr_js, p_tr_js), ("val", y_va_js, p_va_js), ("test", y_te_js, p_te_js)], report_dir, "js")
        plot_accuracy_curve_multi_splits([("train", y_tr_js, p_tr_js), ("val", y_va_js, p_va_js), ("test", y_te_js, p_te_js)], report_dir, "js")

    # Multi-split overlays for Fused: if we have DOM+JS preds, compute fused for train on the fly
    def maybe_fuse(y: np.ndarray, p_dom: np.ndarray, p_js: np.ndarray):
        # Use logistic weights if available
        w_path = os.path.join(args.fusion_dir, "fusion_weights.json")
        if os.path.exists(w_path):
            try:
                w = json.load(open(w_path))
                coef = np.array(w.get("coef"))[0]
                intercept = np.array(w.get("intercept"))[0]
                z = coef[0] * p_dom + coef[1] * p_js + intercept
                pf = 1 / (1 + np.exp(-z))
                return y, pf
            except Exception:
                pass
        # fallback average
        pf = 0.5 * p_dom + 0.5 * p_js
        return y, pf

    if js_train:
        ids_tr_js, y_tr_js, p_tr_js = js_train
        # Align DOM train and JS train by ID for fused overlays
        id2p = {i: p for i, p in zip(ids_tr_js, p_tr_js)}
        dom_ids = [r.get("id") for r in rows_tr]
        idx = [i for i, did in enumerate(dom_ids) if did in id2p]
        if idx:
            y_tr_common = y_tr[idx]
            p_tr_dom_common = p_tr[idx]
            p_tr_js_common = np.array([id2p[dom_ids[i]] for i in idx], dtype=float)
            y_tr_fu, p_tr_fu = maybe_fuse(y_tr_common, p_tr_dom_common, p_tr_js_common)
        else:
            y_tr_fu = p_tr_fu = None
    else:
        y_tr_fu = p_tr_fu = None
    if js_val and fu_val:
        y_va_js, p_va_js = js_val
        y_va_fu, p_va_fu = fu_val
    else:
        y_va_fu = p_va_fu = None
    if js_test and fu_test:
        y_te_js, p_te_js = js_test
        y_te_fu, p_te_fu = fu_test
    else:
        y_te_fu = p_te_fu = None

    if (y_tr_fu is not None) and (y_va_fu is not None) and (y_te_fu is not None):
        y_tr_fu_np = cast(np.ndarray, y_tr_fu)
        p_tr_fu_np = cast(np.ndarray, p_tr_fu)
        y_va_fu_np = cast(np.ndarray, y_va_fu)
        p_va_fu_np = cast(np.ndarray, p_va_fu)
        y_te_fu_np = cast(np.ndarray, y_te_fu)
        p_te_fu_np = cast(np.ndarray, p_te_fu)
        plot_pr_multi_splits([("train", y_tr_fu_np, p_tr_fu_np), ("val", y_va_fu_np, p_va_fu_np), ("test", y_te_fu_np, p_te_fu_np)], report_dir, "fused")
        plot_roc_multi_splits([("train", y_tr_fu_np, p_tr_fu_np), ("val", y_va_fu_np, p_va_fu_np), ("test", y_te_fu_np, p_te_fu_np)], report_dir, "fused")
        plot_reliability_multi_splits([("train", y_tr_fu_np, p_tr_fu_np), ("val", y_va_fu_np, p_va_fu_np), ("test", y_te_fu_np, p_te_fu_np)], report_dir, "fused")
        plot_accuracy_curve_multi_splits([("train", y_tr_fu_np, p_tr_fu_np), ("val", y_va_fu_np, p_va_fu_np), ("test", y_te_fu_np, p_te_fu_np)], report_dir, "fused")

    # Confusion matrices at calibrated thresholds
    plot_confusion(y_va, p_va, thr_95, report_dir, "val_tpr95")
    plot_confusion(y_te, p_te, thr_95, report_dir, "test_tpr95")
    plot_confusion(y_va, p_va, thr_90, report_dir, "val_tpr90")
    plot_confusion(y_te, p_te, thr_90, report_dir, "test_tpr90")

    # If fusion thresholds exist, also plot CMs for fused
    fu_cal = os.path.join(args.fusion_dir, "calibration.json")
    if os.path.exists(fu_cal):
        try:
            with open(fu_cal, "r", encoding="utf-8") as f:
                cal_f = json.load(f)
            thrf_95 = float(cal_f.get("thresholds", {}).get("0.95", {}).get("threshold", thr_95))
            thrf_90 = float(
                cal_f.get("thresholds", {}).get("0.90", {}).get("threshold",
                cal_f.get("thresholds", {}).get("0.9", {}).get("threshold", thr_90))
            )
            if fu_val and fu_test:
                y_va_fu, p_va_fu = fu_val
                y_te_fu, p_te_fu = fu_test
                plot_confusion(y_va_fu, p_va_fu, thrf_95, report_dir, "val_fused_tpr95")
                plot_confusion(y_te_fu, p_te_fu, thrf_95, report_dir, "test_fused_tpr95")
                plot_confusion(y_va_fu, p_va_fu, thrf_90, report_dir, "val_fused_tpr90")
                plot_confusion(y_te_fu, p_te_fu, thrf_90, report_dir, "test_fused_tpr90")
        except Exception:
            pass
    # Meta-fusion confusion matrices if its calibration present
    meta_cal_path = os.path.join(args.meta_fusion_dir, "calibration.json")
    if os.path.exists(meta_cal_path) and meta_val and meta_test:
        try:
            with open(meta_cal_path, "r", encoding="utf-8") as f:
                cal_m = json.load(f)
            thrm_95 = float(cal_m.get("thresholds", {}).get("0.95", {}).get("threshold", thr_95))
            thrm_90 = float(
                cal_m.get("thresholds", {}).get("0.90", {}).get("threshold",
                cal_m.get("thresholds", {}).get("0.9", {}).get("threshold", thr_90))
            )
            y_va_meta, p_va_meta = meta_val
            y_te_meta, p_te_meta = meta_test
            plot_confusion(y_va_meta, p_va_meta, thrm_95, report_dir, "val_meta_tpr95")
            plot_confusion(y_te_meta, p_te_meta, thrm_95, report_dir, "test_meta_tpr95")
            plot_confusion(y_va_meta, p_va_meta, thrm_90, report_dir, "val_meta_tpr90")
            plot_confusion(y_te_meta, p_te_meta, thrm_90, report_dir, "test_meta_tpr90")
        except Exception:
            pass

    # Combined ROC & reliability on test for DOM/JS/Fused
    combined = [("DOM", y_te, p_te)]
    if js_test:
        y_te_js, p_te_js = js_test
        combined.append(("JS", y_te_js, p_te_js))
    if fu_test:
        y_te_fu, p_te_fu = fu_test
        combined.append(("Fused", y_te_fu, p_te_fu))
    if xfu_test:
        y_te_xf, p_te_xf = xfu_test
        combined.append(("XFusion", y_te_xf, p_te_xf))
    if meta_test:
        y_te_meta, p_te_meta = meta_test
        combined.append(("Meta", y_te_meta, p_te_meta))
    if len(combined) > 1:
        plot_roc_multi(combined, report_dir, "test")
        plot_pr_multi(combined, report_dir, "test")
        plot_reliability_multi(combined, report_dir, "test")

    # Training/validation loss curves for DOM and JS (if available)
    plot_train_val_loss(os.path.join(args.model_dir, "trainer_state.json"), report_dir, "dom")
    plot_train_val_loss(os.path.join(args.js_dir, "trainer_state.json"), report_dir, "js")

    # Save a detailed metrics summary and build a pretty HTML table later
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score, average_precision_score

    def metrics_for(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
        """Compute standard metrics robustly, even when y has a single class.
        - log_loss: pass labels=[0,1] to avoid ValueError when y contains a single class
        - roc_auc / pr_auc: return NaN if undefined (single-class y)
        """
        yhat = (p >= thr).astype(int)
        p_clip = np.clip(p, 1e-8, 1 - 1e-8)
        # Always compute well-defined metrics
        acc = float(accuracy_score(y, yhat))
        prec = float(precision_score(y, yhat, zero_division=0))
        rec = float(recall_score(y, yhat, zero_division=0))
        f1 = float(f1_score(y, yhat, zero_division=0))
        # log_loss: specify labels to support single-class y
        try:
            ll = float(log_loss(y, p_clip, labels=[0, 1]))
        except Exception:
            ll = float("nan")
        # AUCs: undefined for single-class y
        try:
            roc = float(roc_auc_score(y, p))
        except Exception:
            roc = float("nan")
        try:
            pr = float(average_precision_score(y, p))
        except Exception:
            pr = float("nan")
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "log_loss": ll,
            "roc_auc": roc,
            "pr_auc": pr,
        }

    metrics_summary: Dict[str, Any] = {
        "thresholds": {"tpr95": thr_95, "tpr90": thr_90},
        "dom": {
            "train@tpr95": metrics_for(y_tr, p_tr, thr_95),
            "val@tpr95": metrics_for(y_va, p_va, thr_95),
            "test@tpr95": metrics_for(y_te, p_te, thr_95),
            "train@tpr90": metrics_for(y_tr, p_tr, thr_90),
            "val@tpr90": metrics_for(y_va, p_va, thr_90),
            "test@tpr90": metrics_for(y_te, p_te, thr_90),
        },
    }
    if js_val and js_test:
        y_va_js, p_va_js = js_val
        y_te_js, p_te_js = js_test
        # Compute pseudo-train metrics if we computed train preds
        if js_train:
            _, y_tr_js, p_tr_js = js_train
            metrics_summary["js"] = {
                "train@tpr95": metrics_for(y_tr_js, p_tr_js, thr_95),
                "val@tpr95": metrics_for(y_va_js, p_va_js, thr_95),
                "test@tpr95": metrics_for(y_te_js, p_te_js, thr_95),
                "train@tpr90": metrics_for(y_tr_js, p_tr_js, thr_90),
                "val@tpr90": metrics_for(y_va_js, p_va_js, thr_90),
                "test@tpr90": metrics_for(y_te_js, p_te_js, thr_90),
            }
        else:
            metrics_summary["js"] = {
                "val@tpr95": metrics_for(y_va_js, p_va_js, thr_95),
                "test@tpr95": metrics_for(y_te_js, p_te_js, thr_95),
                "val@tpr90": metrics_for(y_va_js, p_va_js, thr_90),
                "test@tpr90": metrics_for(y_te_js, p_te_js, thr_90),
            }
    if fu_val and fu_test:
        y_va_fu, p_va_fu = fu_val
        y_te_fu, p_te_fu = fu_test
        # Derive train fused metrics only if we computed fusion above
        if (isinstance(locals().get("p_tr_fu"), np.ndarray)):
            y_tr_fu_np = locals().get("y_tr_fu")
            p_tr_fu_np = locals().get("p_tr_fu")
            if y_tr_fu_np is not None and p_tr_fu_np is not None:
                metrics_summary["fused"] = {
                    "train@tpr95": metrics_for(y_tr_fu_np, p_tr_fu_np, thr_95),
                    "val@tpr95": metrics_for(y_va_fu, p_va_fu, thr_95),
                    "test@tpr95": metrics_for(y_te_fu, p_te_fu, thr_95),
                    "train@tpr90": metrics_for(y_tr_fu_np, p_tr_fu_np, thr_90),
                    "val@tpr90": metrics_for(y_va_fu, p_va_fu, thr_90),
                    "test@tpr90": metrics_for(y_te_fu, p_te_fu, thr_90),
                }
        else:
            metrics_summary["fused"] = {
                "val@tpr95": metrics_for(y_va_fu, p_va_fu, thr_95),
                "test@tpr95": metrics_for(y_te_fu, p_te_fu, thr_95),
                "val@tpr90": metrics_for(y_va_fu, p_va_fu, thr_90),
                "test@tpr90": metrics_for(y_te_fu, p_te_fu, thr_90),
            }
    # Meta-fusion metrics (val/test only)
    if meta_val and meta_test:
        y_va_meta, p_va_meta = meta_val
        y_te_meta, p_te_meta = meta_test
        metrics_summary["meta"] = {
            "val@tpr95": metrics_for(y_va_meta, p_va_meta, thr_95),
            "test@tpr95": metrics_for(y_te_meta, p_te_meta, thr_95),
            "val@tpr90": metrics_for(y_va_meta, p_va_meta, thr_90),
            "test@tpr90": metrics_for(y_te_meta, p_te_meta, thr_90),
        }

    # Write enriched summary including light heads and cascade (if present)
    def _safe_load_json(p: str):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    light_dirs = {
        "url": "artifacts/url_head",
        "js_charcnn": "artifacts/js_charcnn",
        "js_charcnn_aug": "artifacts/js_charcnn_aug",
        "text": "artifacts/text_head",
    }
    # Extend with user-specified head dirs (key = basename)
    if args.heads_dirs:
        for h in args.heads_dirs:
            if not h:
                continue
            name = os.path.basename(h.rstrip("/")) or h
            light_dirs[name] = h
    light_cal: Dict[str, Any] = {}
    for name, d in light_dirs.items():
        cal = _safe_load_json(os.path.join(d, "calibration_eval.json"))
        if cal:
            light_cal[name] = cal
    cascade_base = args.cascade_dir if args.cascade_dir else "artifacts/cascade"
    cascade_json = _safe_load_json(os.path.join(cascade_base, "cascade.json"))

    # Optional deeper cascade analysis: derive per-class resolution using URL + Cheap heads if available
    cascade_analysis: Dict[str, Any] | None = None
    try:
        if cascade_json and isinstance(cascade_json, dict):
            stage1 = cascade_json.get("stage1", {}) or {}
            thr_hi = float(stage1.get("thr_hi", 1.0))
            thr_lo = float(stage1.get("thr_lo", 0.0))
            # Load URL & Cheap predictions for test split to recompute stage1 resolution stats
            def _read_head_preds(head_dir: str, split: str) -> Dict[str, float]:
                out: Dict[str, float] = {}
                try:
                    pth = os.path.join(head_dir, f"preds_{split}.jsonl")
                    if os.path.exists(pth):
                        with open(pth, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                if not line.strip():
                                    continue
                                try:
                                    obj = json.loads(line)
                                except Exception:
                                    continue
                                i = str(obj.get("id"))
                                if i and "prob" in obj:
                                    try:
                                        out[i] = float(obj["prob"])
                                    except Exception:
                                        pass
                except Exception:
                    return {}
                return out
            url_test = _read_head_preds("artifacts/url_head", "test")
            # Align with test rows already loaded earlier (rows_te)
            records: List[Dict[str, Any]] = []
            for r in rows_te:
                rid = r.get("id")
                if rid is None:
                    continue
                sid = str(rid)
                if sid in url_test:
                    try:
                        y_true = int(r.get("label", 0))
                    except Exception:
                        continue
                    s1 = url_test[sid]
                    records.append({"id": sid, "y": y_true, "s1": s1})
            if records:
                import numpy as _np
                s1_scores = _np.array([rec["s1"] for rec in records], dtype=float)
                y_arr = _np.array([rec["y"] for rec in records], dtype=int)
                accept_phish = s1_scores >= thr_hi
                accept_benign = s1_scores <= thr_lo
                resolved = accept_phish | accept_benign
                total = float(len(records))
                pos_mask = y_arr == 1
                neg_mask = y_arr == 0
                pos_total = float(max(1, pos_mask.sum()))
                neg_total = float(max(1, neg_mask.sum()))
                resolved_pos = float((resolved & pos_mask).sum())
                resolved_neg = float((resolved & neg_mask).sum())
                cascade_analysis = {
                    "thr_hi": thr_hi,
                    "thr_lo": thr_lo,
                    "test": {
                        "records": int(total),
                        "resolved_overall_frac": float(resolved.sum() / total if total > 0 else float("nan")),
                        "escalated_overall_frac": float(1.0 - (resolved.sum() / total) if total > 0 else float("nan")),
                        "resolved_phish_frac": float(resolved_pos / pos_total),
                        "resolved_benign_frac": float(resolved_neg / neg_total),
                        "escalated_phish_frac": float(1.0 - (resolved_pos / pos_total)),
                        "escalated_benign_frac": float(1.0 - (resolved_neg / neg_total)),
                        "pos_total": int(pos_mask.sum()),
                        "neg_total": int(neg_mask.sum()),
                    },
                }
    except Exception:
        cascade_analysis = None
    # Attempt to include XFusion (cross-attention fusion) diagnostics if present.
    # Expected layout: <fusion_parent>/fusion_xattn/diagnostics/diagnostics.json
    # We also copy corr_heatmap.png into the report directory for embedding.
    xfusion_diagnostics = None
    try:
        if args.xfusion_diag and os.path.exists(args.xfusion_diag):
            with open(args.xfusion_diag, "r", encoding="utf-8") as f:
                xfusion_diagnostics = json.load(f)
            diag_dir = os.path.dirname(args.xfusion_diag)
        else:
            fusion_parent = os.path.dirname(os.path.abspath(args.fusion_dir)) if os.path.isabs(args.fusion_dir) else os.path.dirname(args.fusion_dir)
            xfusion_dir = args.xfusion_dir if args.xfusion_dir else os.path.join(fusion_parent or "artifacts", "fusion_xattn")
            diag_dir = os.path.join(xfusion_dir, "diagnostics")
            diag_json = os.path.join(diag_dir, "diagnostics.json")
            if os.path.exists(diag_json):
                with open(diag_json, "r", encoding="utf-8") as f:
                    xfusion_diagnostics = json.load(f)
        # Copy correlation heatmap if available
        if xfusion_diagnostics:
            corr_png = os.path.join(diag_dir, "corr_heatmap.png")
            if os.path.exists(corr_png):
                try:
                    shutil.copy2(corr_png, os.path.join(report_dir, "xfusion_corr_heatmap.png"))
                except Exception:
                    pass
    except Exception:
        xfusion_diagnostics = None

    # Macro metrics across core models (focus on test thresholds)
    def _macro_average(models: Dict[str, Any], key: str) -> Dict[str, float]:
        acc: Dict[str, List[float]] = {}
        for mname, sect in models.items():
            if not isinstance(sect, dict):
                continue
            entry = sect.get(key)
            if not isinstance(entry, dict):
                continue
            for mk, mv in entry.items():
                try:
                    fv = float(mv)
                except Exception:
                    continue
                if np.isnan(fv):
                    continue
                acc.setdefault(mk, []).append(fv)
        return {mk: (float(np.mean(vs)) if vs else float("nan")) for mk, vs in acc.items()}

    core_for_macro = {k: metrics_summary.get(k) for k in ["dom","js","fused","meta"] if k in metrics_summary}
    macro_block: Dict[str, Any] = {"models": list(core_for_macro.keys())}
    for split_thr in ["train@tpr95","val@tpr95","test@tpr95","train@tpr90","val@tpr90","test@tpr90"]:
        macro_block[split_thr] = _macro_average(core_for_macro, split_thr)

    # Dataset drift / coverage diagnostics
    dataset_drift: Dict[str, Any] | None = None
    try:
        full_path = args.full_jsonl
        if os.path.exists(full_path):
            # Collect IDs and labels from full dataset
            full_total = 0
            full_pos = 0
            full_ids: set[str] = set()
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f_full:
                for line in f_full:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    fid = obj.get("id")
                    if fid is not None:
                        full_ids.add(str(fid))
                    try:
                        if int(obj.get("label", 0)) == 1:
                            full_pos += 1
                    except Exception:
                        pass
                    full_total += 1
            # Collect IDs used in splits (train/val/test DOM rows already loaded)
            split_ids = set()
            for r in rows_tr + rows_va + rows_te:
                rid = r.get("id")
                if rid is not None:
                    split_ids.add(str(rid))
            missing_ids = sorted(list(full_ids - split_ids))[:50]  # cap preview
            extra_ids = sorted(list(split_ids - full_ids))[:50]
            used_total = len(split_ids)
            dataset_drift = {
                "full_total": full_total,
                "full_pos": full_pos,
                "full_neg": full_total - full_pos,
                "split_total_unique": used_total,
                "coverage_ratio": (used_total / full_total) if full_total > 0 else None,
                "missing_in_splits_example": missing_ids,
                "extra_in_splits_example": extra_ids,
                "missing_count": max(0, len(full_ids) - used_total),
            }
    except Exception as e:
        dataset_drift = {"error": str(e)}

    summary_bundle = {"core": metrics_summary, "light_heads": light_cal, "cascade": cascade_json, "dataset_drift": dataset_drift, "macro": macro_block, "used_splits": used_splits}
    if args.splits_version:
        # If it's a path to v2 file, attempt to parse minimal metadata
        meta = {"declared": args.splits_version}
        if os.path.exists(args.splits_version):
            try:
                with open(args.splits_version, 'r', encoding='utf-8') as sf:
                    js = json.load(sf)
                meta.update({k: js.get(k) for k in ("version","tag","cutoff","hashes","counts") if k in js})
            except Exception as e:
                meta['error'] = str(e)
        summary_bundle['splits'] = meta
    if cascade_analysis is not None:
        summary_bundle["cascade_analysis"] = cascade_analysis
    if xfusion_diagnostics is not None:
        summary_bundle["xfusion_diagnostics"] = xfusion_diagnostics
    with open(os.path.join(report_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_bundle, f, indent=2)

    # Interpretability samples
    if args.lime or args.shap:
        # pick some examples per class from test set for DOM
        pos = [r for r in rows_te if int(r.get("label", 0)) == 1][: args.num_expl]
        neg = [r for r in rows_te if int(r.get("label", 0)) == 0][: args.num_expl]
        samples_dom = pos + neg
        if args.lime:
            explain_lime(
                args.model_dir,
                samples_dom,
                os.path.join(report_dir, "lime_dom"),
                max_length=args.max_length,
                num_features=10,
                num_samples=args.xai_num_samples,
                max_chars=args.xai_max_chars,
                device_preference=args.xai_device,
                tokenizer_mode=args.xai_tokenizer,
            )
        if args.shap:
            explain_shap(
                args.model_dir,
                samples_dom,
                os.path.join(report_dir, "shap_dom"),
                max_length=args.max_length,
                num_samples=args.xai_num_samples,
                max_chars=args.xai_max_chars,
                background_size=args.xai_background,
                device_preference=args.xai_device,
                tokenizer_mode=args.xai_tokenizer,
            )

        # JS samples: build from concatenated script text
        samples_js: List[Dict[str, Any]] = []
        for r in rows_te:
            t = concat_scripts(r)
            if t:
                samples_js.append({"id": r.get("id"), "url": r.get("url"), "text": t, "label": int(r.get("label", 0))})
            if len(samples_js) >= max(1, args.num_expl * 2):
                break
        if samples_js:
            if args.lime:
                explain_lime_js(
                    args.js_dir,
                    samples_js[: args.num_expl * 2],
                    os.path.join(report_dir, "lime_js"),
                    max_length=args.max_length,
                    num_features=10,
                    num_samples=args.xai_num_samples,
                    max_chars=args.xai_max_chars,
                    device_preference=args.xai_device,
                    tokenizer_mode=args.xai_tokenizer,
                )
            if args.shap:
                explain_shap_js(
                    args.js_dir,
                    samples_js[: args.num_expl],
                    os.path.join(report_dir, "shap_js"),
                    max_length=args.max_length,
                    num_samples=args.xai_num_samples,
                    max_chars=args.xai_max_chars,
                    background_size=args.xai_background,
                    device_preference=args.xai_device,
                    tokenizer_mode=args.xai_tokenizer,
                )

    print(f"[INFO] Report written to: {report_dir}")

    # Build an HTML summary page linking key artifacts
    def write_html_index():
        def safe_load(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None

        # Safe float formatter for HTML (avoids formatting None)
        def fmt4(x) -> str:
            try:
                v = float(x)
                import math
                if math.isnan(v) or math.isinf(v):
                    return "n/a"
                return f"{v:.4f}"
            except Exception:
                return "n/a"
        dom_cal = safe_load(os.path.join(args.model_dir, "calibration.json"))
        js_cal = safe_load(os.path.join(args.js_dir, "calibration.json"))
        fu_cal = safe_load(os.path.join(args.fusion_dir, "calibration.json"))
        meta_cal = safe_load(os.path.join(args.meta_fusion_dir, "calibration.json"))
        metrics_json = safe_load(os.path.join(report_dir, "summary.json"))
        pngs = [p for p in os.listdir(report_dir) if p.endswith(".png")]
        used_splits_json = metrics_json.get("used_splits") if isinstance(metrics_json, dict) else None
        used_note = ""
        if isinstance(used_splits_json, dict):
            try:
                u_tr = used_splits_json.get("train")
                u_va = used_splits_json.get("val")
                u_te = used_splits_json.get("test")
                used_note = (
                    "<div class='mono' style='opacity:.75;margin-left:10px'>splits used → "
                    f"train: {escape_html(os.path.basename(u_tr) if u_tr else '')}, "
                    f"val: {escape_html(os.path.basename(u_va) if u_va else '')}, "
                    f"test: {escape_html(os.path.basename(u_te) if u_te else '')}</div>"
                )
            except Exception:
                used_note = ""
        lime_dom_dir = os.path.join(report_dir, "lime_dom")
        shap_dom_dir = os.path.join(report_dir, "shap_dom")
        lime_js_dir = os.path.join(report_dir, "lime_js")
        shap_js_dir = os.path.join(report_dir, "shap_js")
        lime_dom_files = sorted([f for f in os.listdir(lime_dom_dir) if f.endswith(".html")]) if os.path.isdir(lime_dom_dir) else []
        shap_dom_files = sorted([f for f in os.listdir(shap_dom_dir) if f.endswith(".html")]) if os.path.isdir(shap_dom_dir) else []
        lime_js_files = sorted([f for f in os.listdir(lime_js_dir) if f.endswith(".html")]) if os.path.isdir(lime_js_dir) else []
        shap_js_files = sorted([f for f in os.listdir(shap_js_dir) if f.endswith(".html")]) if os.path.isdir(shap_js_dir) else []

        # Dataset summary counts
        def counts(rows: List[Dict[str, Any]]):
            total = len(rows)
            pos = sum(1 for r in rows if int(r.get("label", 0)) == 1)
            neg = total - pos
            return total, neg, pos
        tr_tot, tr_neg, tr_pos = counts(rows_tr)
        va_tot, va_neg, va_pos = counts(rows_va)
        te_tot, te_neg, te_pos = counts(rows_te)
        drift = (metrics_json or {}).get("dataset_drift") if isinstance(metrics_json, dict) else None
        drift_html = ""
        if isinstance(drift, dict) and drift.get("full_total"):
            cov = drift.get("coverage_ratio")
            warn = "" if (cov is None or cov >= 0.98) else "<div class='pill' style='background:#3a1e1e;border:1px solid #6b2a2a;color:#f5b5b5'>WARN: split coverage low</div>"
            if cov is None:
                cov_str = 'n/a'
            else:
                try:
                    cov_str = f"{float(cov):.3f}"
                except Exception:
                    cov_str = 'n/a'
            coverage_line = f"<div class='mono' style='margin-top:4px'>coverage: {cov_str}</div>" + (warn if cov is not None else "")
            drift_html = (
                f"<div class='card'><b>Full Dataset</b><div>{int(drift.get('full_total',0))} docs</div><div>benign: {int(drift.get('full_neg',0))}</div><div>phish: {int(drift.get('full_pos',0))}</div>"
                + coverage_line
                + f"<div class='mono' style='opacity:.7'>missing ids: {int(drift.get('missing_count',0))}</div></div>"
            )
        elif isinstance(drift, dict) and drift.get("error"):
            drift_html = f"<div class='card'><b>Full Dataset</b><div class='mono' style='color:#f88'>error: {drift.get('error')}</div></div>"

        def metric_block(title, cal):
            if not cal:
                return f"<h3>{title}</h3><p>No calibration found.</p>"
            pr = (cal.get("metrics", {}) or {}).get("pr_auc")
            roc = (cal.get("metrics", {}) or {}).get("roc_auc")
            thr = cal.get("thresholds", {}) or {}
            rows = "".join(
                f"<tr><td>{k}</td><td>{fmt4((v or {}).get('threshold'))}</td><td>{fmt4((v or {}).get('fpr'))}</td></tr>"
                for k, v in thr.items()
            )
            return (
                f"<h3>{title}</h3>"
                f"<p>PR-AUC: {fmt4(pr)} | ROC-AUC: {fmt4(roc)}</p>"
                f"<table><thead><tr><th>TPR</th><th>Threshold</th><th>FPR</th></tr></thead><tbody>{rows}</tbody></table>"
            )

        # Build a metrics table if available
        def metrics_table(title: str, section: Dict[str, Any] | None):
            if not section:
                return f"<div class='card'><h3>{title}</h3><p>No metrics.</p></div>"
            keys = sorted(section.keys())
            cols = ["accuracy","precision","recall","f1","log_loss","roc_auc","pr_auc"]
            rows_html = []
            for k in keys:
                m = section.get(k, {}) or {}
                cells = "".join(f"<td class='mono' style='text-align:right'>{fmt4(m.get(c))}</td>" for c in cols)
                rows_html.append(f"<tr><td class='mono'>{k}</td>{cells}</tr>")
            head = "".join(f"<th>{c}</th>" for c in ["split@thr"]+cols)
            return (
                f"<div class='card'><h3>{title}</h3>"
                f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"
            )

        # Tests overview (import dynamically)
        tests_html = ""
        try:
            from tests import test_metrics as _tm  # type: ignore
            from tests import test_calibration as _tc  # type: ignore
            from tests import test_normalize as _tn  # type: ignore
            tests_html = (
                "<div class='card'><h3>Test suites</h3>"
                "<ul>"
                "<li><b>metrics</b>: roc_auc, pr_auc, fpr_at_tpr (sanity checks)</li>"
                "<li><b>calibration</b>: TemperatureScaler reduces log-loss on synthetic data</li>"
                "<li><b>normalize</b>: DOM normalization and script extraction behaviors</li>"
                "</ul>"
                "<div class='mono' style='opacity:.8'>Run: pytest -q (5 tests should pass)</div>"
                "</div>"
            )
        except Exception:
            tests_html = ""

        macro_json = (metrics_json or {}).get("macro") if isinstance(metrics_json, dict) else None
        def macro_table(macro: Dict[str, Any] | None):
            if not isinstance(macro, dict):
                return "<div class='card'><h3>Macro Metrics</h3><p>No macro metrics.</p></div>"
            models = macro.get("models") or []
            split_keys = [k for k in macro.keys() if k != "models"]
            metric_cols: set[str] = set()
            for sk in split_keys:
                metric_cols.update((macro.get(sk) or {}).keys())
            cols = sorted(metric_cols)
            def row(label: str, data: Dict[str, float]):
                cells = "".join(f"<td class='mono' style='text-align:right'>{fmt4(data.get(c))}</td>" for c in cols)
                return f"<tr><td class='mono'>{label}</td>{cells}</tr>"
            head = "".join(f"<th>{c}</th>" for c in ["split@thr"] + cols)
            body = "".join(row(sk, macro.get(sk) or {}) for sk in sorted(split_keys))
            return (
                "<div class='card'><h3>Macro Metrics (average over models: " + ", ".join(models) + ")</h3>"
                + f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
            )

        plot_captions_html = (
            "<div class='card'><h3>Plot Captions</h3>"
            "<ul style='margin:4px 0 0 16px'>"
            "<li><b>ROC Curve</b>: TPR vs FPR; diagonal = random; AUC closer to 1 is better.</li>"
            "<li><b>PR Curve</b>: Precision vs Recall; area (PR-AUC) emphasizes positive class performance under imbalance.</li>"
            "<li><b>Reliability (Calibration)</b>: Predicted probability vs empirical frequency; perfect calibration lies on diagonal.</li>"
            "<li><b>Accuracy vs Threshold</b>: Shows how classification accuracy shifts as decision threshold varies.</li>"
            "<li><b>Confusion Matrix</b>: Counts at given threshold targeting TPR goals (e.g. 0.90/0.95); off‑diagonal cells are errors.</li>"
            "</ul></div>"
        )

        html = [
            "<html><head><meta charset='utf-8'><title>PhisDOM Report</title>",
            "<style>",
            ":root{color-scheme:dark light} body{font-family:system-ui,Arial,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}",
            "a{color:#8ab4f8} img{max-width:1000px;border:1px solid #333;border-radius:6px;margin:8px 0;background:#111}",
            "table{border-collapse:collapse;margin:8px 0;background:#111} td,th{border:1px solid #333;padding:6px 10px}",
            ".pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#1f2330;color:#cbd5e1;border:1px solid #2a2f3a;font-size:12px}",
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}",
            ".card{background:#0f1320;border:1px solid #262b36;border-radius:8px;padding:12px}",
            ".top{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid #262b36;background:#0b0e14;position:sticky;top:0}",
            ".wrap{max-width:1200px;margin:0 auto;padding:16px}",
            ".section{margin-top:16px}",
            ".subgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}",
            ".jump{position:sticky;top:54px;background:#0b0e14;border-bottom:1px solid #262b36;padding:8px 16px;z-index:5}",
            ".jump a{margin-right:10px}",
            "</style>",
            "</head><body>",
            "<div class='top'><div class='pill'>Report</div><div class='mono'>PhisDOM Evaluation</div>"
            f"<div style='margin-left:auto' class='mono'>{args.model_dir}</div>{used_note}</div>",
            "<div class='jump mono'>Jump to: <a href='#dataset'>Dataset</a><a href='#cal'>Calibration</a><a href='#metrics'>Metrics</a><a href='#xfusion'>XFusion</a><a href='#plots'>Plots</a><a href='#xai'>Explanations</a></div>",
            "<div class='wrap'>",
            "<div class='section'><p class='mono' style='opacity:.85'>This report summarizes training/evaluation of the DOM model (MarkupLM) and optionally JS (CodeT5+) and fused heads. It includes PR/ROC curves, reliability, confusion matrices at calibrated thresholds, and optional LIME/SHAP explanations. Below are dataset counts and calibration snapshots, followed by detailed metrics tables and plots. The Tests section summarizes unit tests included in this repo.</p></div>",
            "<div id='dataset' class='section'><h2 style='margin:0 0 8px 0'>Dataset summary</h2>",
            f"<div class='subgrid'><div class='card'><b>Train</b><div>{tr_tot} docs</div><div>benign: {tr_neg}</div><div>phish: {tr_pos}</div></div>",
            f"<div class='card'><b>Val</b><div>{va_tot} docs</div><div>benign: {va_neg}</div><div>phish: {va_pos}</div></div>",
            f"<div class='card'><b>Test</b><div>{te_tot} docs</div><div>benign: {te_neg}</div><div>phish: {te_pos}</div></div></div></div>",
            (drift_html if drift_html else ""),
            "<div id='cal' class='section'><h2 style='margin:0 0 8px 0'>Calibration & Metrics</h2>",
            f"<div class='grid'><div class='card'>{metric_block('DOM (MarkupLM)', dom_cal)}</div>",
            f"<div class='card'>{metric_block('JS (CodeT5+)', js_cal)}</div>",
            f"<div class='card'>{metric_block('Fused (DOM+JS)', fu_cal)}</div>",
            f"<div class='card'>{metric_block('Meta Fused (All Heads)', meta_cal)}</div>",
            # Include light heads if present
            (lambda light: ''.join([
                f"<div class='card'>{metric_block('URL CharCNN', light.get('url'))}</div>",
                f"<div class='card'>{metric_block('Text CharCNN', light.get('text'))}</div>",
                f"<div class='card'>{metric_block('JS CharCNN (base)', light.get('js_charcnn'))}</div>",
                f"<div class='card'>{metric_block('JS CharCNN (aug)', light.get('js_charcnn_aug'))}</div>",
            ]) if isinstance(light, dict) else '')((metrics_json or {}).get('light_heads')),
            "</div></div>",
            "<div id='metrics' class='section'><h2 style='margin:0 0 8px 0'>ML Metrics</h2>",
            metrics_table("DOM", (metrics_json or {}).get("core", {}).get("dom")),
            metrics_table("JS", (metrics_json or {}).get("core", {}).get("js")),
            metrics_table("Fused", (metrics_json or {}).get("core", {}).get("fused")),
            metrics_table("Meta Fused (All Heads)", (metrics_json or {}).get("core", {}).get("meta")),
            macro_table(macro_json),
            "</div>",
        ]

        # XFusion diagnostics card (if present)
        xfd = (metrics_json or {}).get("xfusion_diagnostics") if isinstance(metrics_json, dict) else None
        if isinstance(xfd, dict):
            try:
                top_grad = xfd.get("top_grad_params") or []
                modalities = xfd.get("modalities") or {}
                corr = xfd.get("correlation_matrix") or []
                corr_mods = xfd.get("correlation_modalities") or []
                steps = xfd.get("total_steps")
                # Build gradient list
                grad_rows = []
                for name, val in top_grad[:10]:
                    try:
                        grad_rows.append(f"<tr><td class='mono'>{name}</td><td class='mono' style='text-align:right'>{float(val):.4f}</td></tr>")
                    except Exception:
                        continue
                grad_tbl = ("<table><thead><tr><th>Param</th><th>|grad|</th></tr></thead><tbody>"+"".join(grad_rows)+"</tbody></table>") if grad_rows else "<p>No gradient stats.</p>"
                # Modalities stats
                mod_cards = []
                for m, stats in modalities.items():
                    if not isinstance(stats, dict):
                        continue
                    mean_n = stats.get("mean_token_norm_mean")
                    var_n = stats.get("mean_token_norm_var")
                    ntok = stats.get("num_tokens")
                    if mean_n is None or var_n is None or ntok is None:
                        continue
                    try:
                        mean_f = float(mean_n)
                        var_f = float(var_n)
                        ntok_i = int(ntok)
                        mod_cards.append(
                            f"<div class='card'><h4 style='margin:0 0 6px 0'>{m}</h4><div class='mono'>mean_norm={mean_f:.4f}</div><div class='mono'>var_norm={var_f:.4f}</div><div class='mono'>tokens={ntok_i}</div></div>"
                        )
                    except Exception:
                        continue
                # Correlation matrix
                corr_html = ""
                if corr and corr_mods and all(isinstance(r, (list, tuple)) for r in corr):
                    try:
                        head = "".join(f"<th>{m}</th>" for m in corr_mods)
                        body_rows = []
                        for mi, row in zip(corr_mods, corr):
                            cells = []
                            for v in row:
                                try:
                                    cells.append(f"<td class='mono' style='text-align:right'>{float(v):.3f}</td>")
                                except Exception:
                                    cells.append("<td class='mono'>n/a</td>")
                            body_rows.append(f"<tr><td class='mono'>{mi}</td>{''.join(cells)}</tr>")
                        corr_html = (
                            "<div style='overflow:auto'><table><thead><tr><th></th>"+head+"</tr></thead><tbody>"+"".join(body_rows)+"</tbody></table></div>"
                        )
                    except Exception:
                        corr_html = "<p>Correlation matrix parse error.</p>"
                heat_png = "xfusion_corr_heatmap.png"
                heat_img = f"<div><img src='{heat_png}' alt='corr heatmap'/></div>" if os.path.exists(os.path.join(report_dir, heat_png)) else ""
                # Attention entropy card (with optional trend + alerts)
                attn_entropy_html = ""
                attn_list = xfd.get('attn_entropy') if isinstance(xfd, dict) else None
                trend = xfd.get('attn_entropy_trend') if isinstance(xfd, dict) else None
                alerts_list = xfd.get('attn_entropy_alerts') if isinstance(xfd, dict) else None
                trend_img = "attn_entropy_trend.png"
                trend_block = ""
                if os.path.exists(os.path.join(report_dir, trend_img)):
                    trend_block = f"<div style='margin-top:6px'><img src='{trend_img}' alt='entropy trend'/></div>"
                alert_block = ""
                if isinstance(alerts_list, list) and alerts_list:
                    items = "".join(f"<li class='mono'>{escape_html(str(a))}</li>" for a in alerts_list)
                    alert_block = f"<div style='margin-top:6px'><strong>Alerts:</strong><ul style='margin:4px 0 0 16px'>{items}</ul></div>"
                if isinstance(attn_list, list):
                    rows = []
                    for e in attn_list:
                        if not isinstance(e, dict):
                            continue
                        if 'mean_entropy' in e:
                            val = e.get('mean_entropy')
                            mean_e = float(val) if isinstance(val, (int, float)) else float('nan')
                            heads = e.get('head_mean_entropy', []) or []
                            head_str = ", ".join(
                                f"{float(h):.2f}" for h in heads if isinstance(h, (int, float))
                            )
                            rows.append(
                                f"<tr><td class='mono'>{e.get('layer')}</td><td class='mono' style='text-align:right'>{mean_e:.3f}</td><td class='mono' style='text-align:right'>{len(heads)}</td><td class='mono'>{head_str}</td></tr>"
                            )
                        elif 'skipped' in e:
                            rows.append(
                                f"<tr><td class='mono'>{e.get('layer')}</td><td colspan='3' class='mono'>skipped (T={e.get('T','?')})</td></tr>"
                            )
                    table = (
                        "<table><thead><tr><th>Layer</th><th>Mean</th><th>Heads</th><th>Head Means</th></tr></thead><tbody>"
                        + "".join(rows)
                        + "</tbody></table>"
                    ) if rows else "<p>No entropy recorded.</p>"
                    meta_line = ""
                    if isinstance(trend, dict):
                        try:
                            nsteps = len(trend.get('steps', []) or [])
                            meta_line = f"<p class='mono' style='opacity:.7;margin:4px 0 0 0'>snapshots={nsteps}</p>"
                        except Exception:
                            pass
                    attn_entropy_html = (
                        f"<div class='card'><h3 style='margin:0 0 6px 0'>Attention Entropy</h3>{table}{meta_line}{trend_block}{alert_block}</div>"
                    )
                    html.append(
                        "<div id='xfusion' class='section'><h2 style='margin:0 0 8px 0'>XFusion Diagnostics</h2>"
                        + f"<p class='mono' style='opacity:.8'>Instrumented cross-attention fusion stats (steps={steps}).</p>"
                        + "<div class='grid'>"
                        + f"<div class='card'><h3 style='margin:0 0 6px 0'>Top Grad Norms</h3>{grad_tbl}</div>"
                        + f"<div class='card'><h3 style='margin:0 0 6px 0'>Modality Token Norms</h3><div class='subgrid'>{''.join(mod_cards) or '<p>No modality stats.</p>'}</div></div>"
                        + f"<div class='card'><h3 style='margin:0 0 6px 0'>Embedding Correlations</h3>{corr_html}{heat_img}</div>"
                        + attn_entropy_html
                        + "</div></div>"
                    )
            except Exception:
                pass
        html.extend([
            "<div id='tests' class='section'><h2 style='margin:0 0 8px 0'>Tests overview</h2>", tests_html, "</div>",
            "<div id='plots' class='section'><h2 style='margin:0 0 8px 0'>Key Plots</h2>",
        ])

        if plot_captions_html:
            html.append(plot_captions_html)
        for name in sorted(pngs):
            html.append(f"<div><p class='mono' style='opacity:.75'>{name}</p><img src='{name}'/></div>")

        if lime_dom_files or shap_dom_files or lime_js_files or shap_js_files:
            html.append("<div id='xai' class='section'><h2 style='margin:0 0 8px 0'>Explanations</h2><div class='grid'>")
            # Build id->row map for test set to retrieve original URLs
            id2row = {str(r.get("id")): r for r in rows_te if r.get("id") is not None}

            if lime_dom_files:
                html.append("<div class='card'><h3>LIME (DOM)</h3><ul>")
                for f in lime_dom_files:
                    rid = f.replace("lime_", "").replace(".html", "")
                    url = id2row.get(rid, {}).get("url") if rid in id2row else None
                    url_html = f" — <a href='{url}' target='_blank'>visit</a>" if url else ""
                    html.append(f"<li><a href='lime_dom/{f}' target='_blank'>{f}</a>{url_html}</li>")
                html.append("</ul></div>")
            if shap_dom_files:
                html.append("<div class='card'><h3>SHAP (DOM)</h3><ul>")
                for f in shap_dom_files:
                    rid = f.replace("shap_", "").replace(".html", "")
                    url = id2row.get(rid, {}).get("url") if rid in id2row else None
                    url_html = f" — <a href='{url}' target='_blank'>visit</a>" if url else ""
                    html.append(f"<li><a href='shap_dom/{f}' target='_blank'>{f}</a>{url_html}</li>")
                html.append("</ul></div>")
            if lime_js_files:
                html.append("<div class='card'><h3>LIME (JS)</h3><ul>")
                for f in lime_js_files:
                    rid = f.replace("lime_js_", "").replace(".html", "")
                    url = id2row.get(rid, {}).get("url") if rid in id2row else None
                    url_html = f" — <a href='{url}' target='_blank'>visit</a>" if url else ""
                    html.append(f"<li><a href='lime_js/{f}' target='_blank'>{f}</a>{url_html}</li>")
                html.append("</ul></div>")
            if shap_js_files:
                html.append("<div class='card'><h3>SHAP (JS)</h3><ul>")
                for f in shap_js_files:
                    rid = f.replace("shap_js_", "").replace(".html", "")
                    url = id2row.get(rid, {}).get("url") if rid in id2row else None
                    url_html = f" — <a href='{url}' target='_blank'>visit</a>" if url else ""
                    html.append(f"<li><a href='shap_js/{f}' target='_blank'>{f}</a>{url_html}</li>")
                html.append("</ul></div>")
            html.append("</div></div>")

        # Cascade coverage (if available)
        cas = (metrics_json or {}).get('cascade')
        cas_analysis = (metrics_json or {}).get('cascade_analysis')
        if cas and isinstance(cas, dict):
            cov = cas.get('coverage', {}) or {}
            s1 = cas.get('stage1', {}) or {}
            rows_cov = []
            for split in ['val','test']:
                c = cov.get(split, {}) or {}
                try:
                    rows_cov.append(f"<tr><td class='mono'>{split}</td><td class='mono' style='text-align:right'>{float(c.get('overall', float('nan'))):.3f}</td><td class='mono' style='text-align:right'>{float(c.get('phish', float('nan'))):.3f}</td><td class='mono' style='text-align:right'>{float(c.get('benign', float('nan'))):.3f}</td></tr>")
                except Exception:
                    continue
            html.append("<div id='cascade' class='section'><h2 style='margin:0 0 8px 0'>Cascade Coverage</h2><div class='card'>")
            html.append("<table><thead><tr><th>split</th><th>overall</th><th>phish</th><th>benign</th></tr></thead><tbody>"+"".join(rows_cov)+"</tbody></table>")
            try:
                html.append(f"<div class='mono' style='opacity:.85'>stage1 thr_hi={float(s1.get('thr_hi', float('nan'))):.4f}, thr_lo={float(s1.get('thr_lo', float('nan'))):.4f}</div>")
            except Exception:
                pass
            # Narrative & efficiency gain (test split)
            if cas_analysis and isinstance(cas_analysis, dict):
                test_block = cas_analysis.get('test') or {}
                try:
                    eff = float(test_block.get('resolved_overall_frac', float('nan')))
                    esc = float(test_block.get('escalated_overall_frac', float('nan')))
                    r_pos = float(test_block.get('resolved_phish_frac', float('nan')))
                    r_neg = float(test_block.get('resolved_benign_frac', float('nan')))
                    html.append(
                        "<div style='margin-top:6px;font-size:13px;line-height:1.35'>"
                        f"Stage-1 quickly resolves <b>{eff:.1%}</b> of test pages (efficiency gain), leaving <b>{esc:.1%}</b> to the heavier fused model. "
                        f"Resolution is class-skewed: <b>{r_pos:.1%}</b> of phishing and <b>{r_neg:.1%}</b> of benign samples decided early. "
                        "`efficiency gain = resolved_overall_frac = coverage of stage-1 decisions`; the remaining band is intentionally narrow to protect recall."
                        "</div>"
                    )
                except Exception:
                    pass
            html.append("</div></div>")

        # Robustness delta card: JS base vs augmented (if both present)
        def _rob_card(light):
            if not isinstance(light, dict):
                return ""
            base = light.get('js_charcnn') or {}
            aug = light.get('js_charcnn_aug') or {}
            pr_b = _safe_float((base.get('metrics') or {}).get('pr_auc'))
            pr_a = _safe_float((aug.get('metrics') or {}).get('pr_auc'))
            roc_b = _safe_float((base.get('metrics') or {}).get('roc_auc'))
            roc_a = _safe_float((aug.get('metrics') or {}).get('roc_auc'))
            # If any metric is missing/unparseable, skip the card (preserve prior behavior)
            if any(np.isnan(v) for v in (pr_b, pr_a, roc_b, roc_a)):
                return ""
            d_pr = pr_a - pr_b
            d_roc = roc_a - roc_b
            sign = lambda v: "+" if v >= 0 else ""
            return (
                "<div id='robust' class='section'><h2 style='margin:0 0 8px 0'>Robustness: JS Augmentation</h2>"
                "<div class='grid'><div class='card'>"
                f"<h3>JS CharCNN (aug vs base)</h3>"
                f"<div class='mono'>PR-AUC: {pr_b:.4f} → {pr_a:.4f} (<b>{sign(d_pr)}{d_pr:.4f}</b>)</div>"
                f"<div class='mono'>ROC-AUC: {roc_b:.4f} → {roc_a:.4f} (<b>{sign(d_roc)}{d_roc:.4f}</b>)</div>"
                "</div></div></div>"
            )
        html.append(_rob_card((metrics_json or {}).get('light_heads')))

        html.append("</div></body></html>")
        with open(os.path.join(report_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        print(f"[INFO] HTML report: {os.path.join(report_dir, 'index.html')}")

    write_html_index()

    # Optional: darkify standalone explanation HTMLs for consistency
    def _darkify_html_dir(d: str):
        if not os.path.isdir(d):
            return
        css = (
            "<style>body{background:#0f1115;color:#e6e6e6;font-family:system-ui,Arial,sans-serif}"
            "a{color:#8ab4f8} .lime{color:#e6e6e6} .table{background:#111} svg{background:#0f1115}"
            "</style>"
        )
        for name in os.listdir(d):
            if not name.endswith(".html"):
                continue
            p = os.path.join(d, name)
            try:
                s = open(p, "r", encoding="utf-8", errors="ignore").read()
                if "color-scheme:dark" in s:
                    continue
                if "<head>" in s:
                    s = s.replace("<head>", "<head><meta name='color-scheme' content='dark'>" + css, 1)
                else:
                    s = "<meta name='color-scheme' content='dark'>" + css + s
                open(p, "w", encoding="utf-8").write(s)
            except Exception:
                pass

    _darkify_html_dir(os.path.join(report_dir, "lime_dom"))
    _darkify_html_dir(os.path.join(report_dir, "shap_dom"))
    _darkify_html_dir(os.path.join(report_dir, "lime_js"))
    _darkify_html_dir(os.path.join(report_dir, "shap_js"))


if __name__ == "__main__":
    main()
