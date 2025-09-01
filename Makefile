# Simple automation for fetching feeds, unifying seed, crawling, training, and evaluation

PY := PYTHONPATH=src python
SEED ?=
LIMIT_PHISH ?= 500
LIMIT_BENIGN ?= 500
AUTO_CUTOFF ?= 80
VAL_FRAC ?= 0.1
CRAWL_CONCURRENCY ?= 12
CRAWL_TIMEOUT ?= 2.0
OUTDIR ?= artifacts/markup_run
MAXLEN ?= 512
BATCH ?= 4
EPOCHS ?= 1
LR ?= 3e-5
XAI_DEVICE ?= cuda

# Ensure env for PhishTank
# export PHISHTANK_APP_KEY=your_key
# export PHISHTANK_UA="phishtank/your-ua"

.PHONY: feeds tranco fetch unify crawl splits slice train eval all e2e clean

feeds: tranco
	$(PY) scripts/fetch_feeds.py --config configs/feeds.yaml

# Generate Tranco CSV locally (requires `pip install tranco`)
tranco:
	@mkdir -p data/feeds
	$(PY) scripts/gen_tranco.py

unify:
	@mkdir -p data
	$(PY) scripts/unify_feeds.py \
		--openphish data/feeds/openphish.txt \
		--phishtank data/feeds/phishtank.csv \
		--tranco data/feeds/tranco.csv \
		--out data/seed.csv \
		--limit-phish $(LIMIT_PHISH) --limit-benign $(LIMIT_BENIGN) \
		--shuffle $(if $(SEED),--seed $(SEED),)

crawl:
	@mkdir -p data
	$(PY) scripts/crawl_playwright.py --input-csv data/seed.csv --out-jsonl data/pages.jsonl --concurrency $(CRAWL_CONCURRENCY) --timeout-s $(CRAWL_TIMEOUT) --block-assets --no-external-js

splits:
	$(PY) scripts/make_splits.py --dataset data/pages.jsonl --out data/splits.json --auto-cutoff-percentile $(AUTO_CUTOFF) --val-frac $(VAL_FRAC) $(if $(SEED),--seed $(SEED),)

slice:
	$(PY) scripts/slice_dataset.py --dataset data/pages.jsonl --splits data/splits.json --out-dir data

train:
	$(PY) scripts/train_markup.py --config configs/markup_base.yaml

# Uses the trained model dir from markup_base.yaml
# Adjust --max-length if needed

# Eval assumes model saved to $(OUTDIR)
# (markup_base.yaml default is artifacts/markup_run)

eval:
	$(PY) scripts/eval_markup.py --model-dir $(OUTDIR) \
		--val-jsonl data/pages_val.jsonl \
		--test-jsonl data/pages_test.jsonl \
		--max-length $(MAXLEN)

.PHONY: train-js eval-js
train-js:
	$(PY) scripts/train_js_codet5p.py --train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/js_codet5p --model-name Salesforce/codet5p-220m --max-length 512 --batch-size 4 --num-epochs 1 --lr 3e-5

eval-js:
	$(PY) scripts/eval_js_codet5p.py --model-dir artifacts/js_codet5p --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --max-length 512

.PHONY: fuse
fuse:
	$(PY) scripts/fuse_heads.py --dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --out-dir artifacts/fusion --method logistic

.PHONY: report
report:
	$(PY) scripts/report_eval.py --model-dir $(OUTDIR) --train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --max-length $(MAXLEN)

.PHONY: report-xai
report-xai:
	$(PY) scripts/report_eval.py --model-dir $(OUTDIR) \
		--train-jsonl data/pages_train.jsonl \
		--val-jsonl data/pages_val.jsonl \
		--test-jsonl data/pages_test.jsonl \
		--max-length $(MAXLEN) \
		--lime --shap --num-expl 1 \
		--xai-device $(XAI_DEVICE) --xai-max-chars 1500 --xai-num-samples 150 --xai-background 3

# End-to-end: fetch feeds, unify to seed, crawl, split, train, eval
all e2e: feeds unify crawl splits slice train eval report train-js eval-js fuse report report-xai

# Resume from splits onward
.PHONY: resume
resume: splits slice train eval

clean:
	rm -f data/pages.jsonl data/pages_train.jsonl data/pages_val.jsonl data/pages_test.jsonl data/splits.json
	rm -rf artifacts/markup_run
