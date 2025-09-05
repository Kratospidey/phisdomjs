# Simple automation for fetching feeds, unifying seed, crawling, training, and evaluation

PY := PYTHONPATH=src python
SEED ?=
LIMIT_PHISH ?= 5000
LIMIT_BENIGN ?= 5000
# Single target size (overrides per-source limits when set)
SEED_SIZE ?=
PHISH_RATIO ?= 0.5
AUTO_CUTOFF ?= 80
VAL_FRAC ?= 0.1
CRAWL_CONCURRENCY ?= 12
CRAWL_TIMEOUT ?= 3.0
CRAWL_RETRIES ?= 1
# Control whether to run the crawler (set to false/0/no to skip and use existing data/pages.jsonl)
CRAWL ?= true
# Whether to include external datasets (e.g., MTLP) alongside internal pages.jsonl
EXTERNAL ?= false
# Paths for external dataset integration
EXTERNAL_DIR ?= data/external/MTLP_Dataset
EXTERNAL_CSV ?= $(EXTERNAL_DIR)/MTLP_Dataset.csv
EXTERNAL_JSONL ?= data/external/mtlp.jsonl
# Final dataset path used for splits/slice
DATASET := $(if $(filter true 1 yes,$(EXTERNAL)),data/pages_merged.jsonl,data/pages.jsonl)
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
	$(PY) scripts/gen_tranco.py $(if $(TRANCO_DATE),--date $(TRANCO_DATE),) $(if $(TRANCO_LIST_ID),--list-id $(TRANCO_LIST_ID),) $(if $(TRANCO_SUBDOMAINS),--subdomains,) $(if $(TRANCO_COUNT),--count $(TRANCO_COUNT),)

unify:
	@mkdir -p data
	$(PY) scripts/unify_feeds.py \
		--openphish data/feeds/openphish.txt \
		--phishtank data/feeds/phishtank.csv \
		--tranco data/feeds/tranco.csv \
		--out data/seed.csv \
		$(if $(SEED_SIZE),--target-size $(SEED_SIZE) --phish-ratio $(PHISH_RATIO),--limit-phish $(LIMIT_PHISH) --limit-benign $(LIMIT_BENIGN)) \
		--shuffle $(if $(SEED),--seed $(SEED),)

crawl:
	@mkdir -p data
	@if [ "$(CRAWL)" = "false" ] || [ "$(CRAWL)" = "0" ] || [ "$(CRAWL)" = "no" ]; then \
		echo "[MAKE] Skipping crawl (CRAWL=$(CRAWL))"; \
		if [ ! -s data/pages.jsonl ]; then \
			echo "[MAKE][ERROR] data/pages.jsonl is missing. Cannot skip crawl."; \
			exit 2; \
		fi; \
	else \
		$(PY) scripts/crawl_playwright.py --input-csv data/seed.csv --out-jsonl data/pages.jsonl --concurrency $(CRAWL_CONCURRENCY) --timeout-s $(CRAWL_TIMEOUT) --block-assets --no-external-js --retries $(CRAWL_RETRIES); \
	fi

.PHONY: import-external
import-external:
	@if [ ! -s "$(EXTERNAL_CSV)" ]; then \
		echo "[MAKE][WARN] External CSV not found at $(EXTERNAL_CSV); skipping import"; \
	else \
		$(PY) scripts/import_mtlp.py --csv $(EXTERNAL_CSV) --out $(EXTERNAL_JSONL); \
	fi

.PHONY: merge
merge:
	@mkdir -p data
	@if [ ! -s "$(EXTERNAL_JSONL)" ]; then \
		echo "[MAKE][ERROR] External JSONL $(EXTERNAL_JSONL) missing; run 'make import-external' or set EXTERNAL=false"; \
		exit 2; \
	fi
	@if [ ! -s data/pages.jsonl ]; then \
		echo "[MAKE][ERROR] data/pages.jsonl missing; generate or set CRAWL=false only if it exists"; \
		exit 2; \
	fi
	$(PY) scripts/merge_datasets.py --inputs data/pages.jsonl $(EXTERNAL_JSONL) --out data/pages_merged.jsonl

splits:
	$(PY) scripts/make_splits.py --dataset $(DATASET) --out data/splits.json --auto-cutoff-percentile $(AUTO_CUTOFF) --val-frac $(VAL_FRAC) $(if $(SEED),--seed $(SEED),)

slice:
	$(PY) scripts/slice_dataset.py --dataset $(DATASET) --splits data/splits.json --out-dir data

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

# End-to-end: fetch feeds, unify to seed, optional crawl, then splits/train/eval
ifeq ($(CRAWL),false)
ifeq ($(EXTERNAL),true)
all e2e: feeds unify crawl-verify import-external merge splits slice train eval report train-js eval-js fuse report report-xai
else ifeq ($(EXTERNAL),1)
all e2e: feeds unify crawl-verify import-external merge splits slice train eval report train-js eval-js fuse report report-xai
else ifeq ($(EXTERNAL),yes)
all e2e: feeds unify crawl-verify import-external merge splits slice train eval report train-js eval-js fuse report report-xai
else
all e2e: feeds unify crawl-verify splits slice train eval report train-js eval-js fuse report report-xai
endif
else ifeq ($(CRAWL),0)
ifeq ($(EXTERNAL),true)
all e2e: feeds unify crawl-verify import-external merge splits slice train eval report train-js eval-js fuse report report-xai
else ifeq ($(EXTERNAL),1)
all e2e: feeds unify crawl-verify import-external merge splits slice train eval report train-js eval-js fuse report report-xai
else ifeq ($(EXTERNAL),yes)
all e2e: feeds unify crawl-verify import-external merge splits slice train eval report train-js eval-js fuse report report-xai
else
all e2e: feeds unify crawl-verify splits slice train eval report train-js eval-js fuse report report-xai
endif
else ifeq ($(CRAWL),no)
ifeq ($(EXTERNAL),true)
all e2e: feeds unify crawl-verify import-external merge splits slice train eval report train-js eval-js fuse report report-xai
else ifeq ($(EXTERNAL),1)
all e2e: feeds unify crawl-verify import-external merge splits slice train eval report train-js eval-js fuse report report-xai
else ifeq ($(EXTERNAL),yes)
all e2e: feeds unify crawl-verify import-external merge splits slice train eval report train-js eval-js fuse report report-xai
else
all e2e: feeds unify crawl-verify splits slice train eval report train-js eval-js fuse report report-xai
endif
else
ifeq ($(EXTERNAL),true)
all e2e: feeds unify crawl import-external merge splits slice train eval report train-js eval-js fuse report report-xai
else ifeq ($(EXTERNAL),1)
all e2e: feeds unify crawl import-external merge splits slice train eval report train-js eval-js fuse report report-xai
else ifeq ($(EXTERNAL),yes)
all e2e: feeds unify crawl import-external merge splits slice train eval report train-js eval-js fuse report report-xai
else
all e2e: feeds unify crawl splits slice train eval report train-js eval-js fuse report report-xai
endif
endif

.PHONY: crawl-verify
crawl-verify:
	@if [ ! -s data/pages.jsonl ]; then \
		echo "[MAKE][ERROR] data/pages.jsonl missing; set CRAWL=true to generate it."; \
		exit 2; \
	else \
		echo "[MAKE] Using existing data/pages.jsonl (skipped crawl)"; \
	fi

# Resume from splits onward
.PHONY: resume
resume: splits slice train eval

clean:
	rm -f data/pages.jsonl data/pages_train.jsonl data/pages_val.jsonl data/pages_test.jsonl data/splits.json
	rm -rf artifacts/markup_run
