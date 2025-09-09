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
# Class balance targets (positive ratios) and tolerance for splits
POSR_TRAIN ?=
POSR_VAL ?=
POSR_TEST ?=
RATIO_TOL ?= 0.05
BALANCE_SPLITS ?= 0
MIN_TOTAL_TEST ?=
CRAWL_CONCURRENCY ?= 12
CRAWL_TIMEOUT ?= 3.0
CRAWL_RETRIES ?= 1
# New lightweight feature toggles
TLS_TIMEOUT ?= 3.0
DNS_TIMEOUT ?= 2.0
MOBILE_PROFILE ?= false
GPU ?= false
# Backfill network lookups toggle (1=true/0=false)
BACKFILL_NETWORK ?= 1
BACKFILL_WORKERS ?= 4
BACKFILL_BATCH ?= 4000
BACKFILL_DISABLE_VISUALS ?= 1
BACKFILL_MAX_IMAGE_BYTES ?= 262144
BACKFILL_DROP_RAW ?= 1
# Phase 6 (augmentation) toggle (1=true/0=false)
AUGMENT_JS ?= 0
# Control whether to run the crawler (set to false/0/no to skip and use existing data/pages.jsonl)
CRAWL ?= true
# Final dataset path used for splits/slice (feeds-only workflow)
DATASET := data/pages.jsonl
OUTDIR ?= artifacts/markup_run
MAXLEN ?= 512
BATCH ?= 4
EPOCHS ?= 10
LR ?= 3e-5
XAI_DEVICE ?= cuda
# Markup eval HTML truncation controls (disabled by default)
MARKUP_TRUNCATE ?= 0              # 0 = no truncation (pass --max-html-chars -1)
MARKUP_MAX_HTML_CHARS ?= 800000    # Used only when MARKUP_TRUNCATE=1
# Early stopping and logging controls
ES_PATIENCE ?= 3
ES_MIN_DELTA ?= 0.0
DISABLE_TQDM ?= 0
SMOKE ?= 0
XF_NO_URL ?= 0
XF_NO_JS ?= 0
XF_NO_TEXT ?= 1
XF_NO_DOM ?= 0
XF_NO_CHEAP ?= 0
XF_JS_RAW_FIELD ?=
XF_NO_JS_CANON ?= 0
XF_HTML_CANON ?= 0
XF_HTML_FIELD ?= html
JS_HEAD_NUM_WORKERS ?= 4

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

# New variable for automatic URL target checking
MIN_CRAWLED_URLS ?= 14750
PHISH_RATIO ?= 0.5

# Check if we need more URLs and crawl incrementally if needed
.PHONY: ensure-crawled-count
ensure-crawled-count:
	@current_count=$$(if [ -f data/pages.jsonl ]; then wc -l < data/pages.jsonl; else echo 0; fi); \
	echo "[MAKE] Current crawled URLs: $$current_count"; \
	echo "[MAKE] Minimum required URLs: $(MIN_CRAWLED_URLS)"; \
	if [ $$current_count -lt $(MIN_CRAWLED_URLS) ]; then \
		echo "[MAKE] Need more URLs - running incremental crawler..."; \
		$(MAKE) crawl-incremental TARGET_URLS=$(MIN_CRAWLED_URLS); \
		echo "[MAKE] Updating splits to include new URLs..."; \
		$(MAKE) extend-splits-if-needed; \
	else \
		echo "[MAKE] URL count sufficient ($$current_count >= $(MIN_CRAWLED_URLS))"; \
		$(MAKE) extend-splits-if-needed; \
	fi

# Extend splits to include any new URLs that aren't covered by existing splits
.PHONY: extend-splits-if-needed
extend-splits-if-needed:
	@if [ -f data/splits.json ] && [ -f data/pages.jsonl ]; then \
		echo "[MAKE] Checking if splits need extension..."; \
		$(PY) scripts/extend_splits.py --dataset data/pages.jsonl --splits data/splits.json --out data/splits_full.json --slice-out-dir data; \
		if [ -f data/splits_full.json ]; then \
			echo "[MAKE] Extended splits and slice files created"; \
		fi; \
	else \
		echo "[MAKE] No existing splits or dataset found - will be created in splits target"; \
	fi

# Incremental crawling using mixed sources (Tranco + OpenPhish/PhishTank)
.PHONY: crawl-incremental
crawl-incremental:
	@if [ "$(CRAWL)" = "false" ] || [ "$(CRAWL)" = "0" ] || [ "$(CRAWL)" = "no" ]; then \
		echo "[MAKE] Incremental crawl skipped (CRAWL=$(CRAWL))"; \
	else \
		echo "[MAKE] Running incremental mixed crawler ($(PHISH_RATIO) phish ratio)..."; \
		bash -c "source ~/.bashrc && conda activate mlcuda311 && PYTHONPATH=src python scripts/crawl_mixed_incremental.py \
			--target-count $(if $(TARGET_URLS),$(TARGET_URLS),$(MIN_CRAWLED_URLS)) \
			--phish-ratio $(PHISH_RATIO) \
			--concurrency $(CRAWL_CONCURRENCY) \
			--timeout-s $(CRAWL_TIMEOUT) \
			--retries $(CRAWL_RETRIES) \
			--tls-timeout $(TLS_TIMEOUT) \
			--dns-timeout $(DNS_TIMEOUT)"; \
	fi

crawl:
	@mkdir -p data
	@if [ "$(CRAWL)" = "false" ] || [ "$(CRAWL)" = "0" ] || [ "$(CRAWL)" = "no" ]; then \
		echo "[MAKE] Skipping crawl (CRAWL=$(CRAWL))"; \
		if [ ! -s data/pages.jsonl ]; then \
			echo "[MAKE][ERROR] data/pages.jsonl is missing. Cannot skip crawl."; \
			exit 2; \
		fi; \
	else \
		$(PY) scripts/crawl_playwright.py --input-csv data/seed.csv --out-jsonl data/pages.jsonl --concurrency $(CRAWL_CONCURRENCY) --timeout-s $(CRAWL_TIMEOUT) --block-assets --no-external-js --retries $(CRAWL_RETRIES) --tls-timeout $(TLS_TIMEOUT) --dns-timeout $(DNS_TIMEOUT) $(if $(filter $(MOBILE_PROFILE),true),--mobile-profile,) $(if $(filter $(GPU),true),--gpu,); \
	fi

splits:
	$(PY) scripts/make_splits.py --dataset $(DATASET) --out data/splits.json --auto-cutoff-percentile $(AUTO_CUTOFF) --val-frac $(VAL_FRAC) \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(POSR_TRAIN),--target-pos-ratio-train $(POSR_TRAIN),) \
		$(if $(POSR_VAL),--target-pos-ratio-val $(POSR_VAL),) \
		$(if $(POSR_TEST),--target-pos-ratio-test $(POSR_TEST),) \
		$(if $(RATIO_TOL),--ratio-tol $(RATIO_TOL),) \
		$(if $(MIN_TOTAL_TEST),--min-total-test $(MIN_TOTAL_TEST),) \
		$(if $(filter $(BALANCE_SPLITS),1),--balance-splits,)

slice:
	$(PY) scripts/slice_dataset.py --dataset $(DATASET) --splits data/splits.json --out-dir data
	$(PY) scripts/verify_splits.py --train data/pages_train.jsonl --val data/pages_val.jsonl --test data/pages_test.jsonl || (echo "[MAKE][ERROR] Bad splits; aborting." && exit 3)

# Aliases for convenience/compatibility with some docs/logs
.PHONY: make_splits slice_dataset
make_splits: splits
slice_dataset: slice

train:
	$(PY) scripts/train_markup.py --config configs/markup_base.yaml \
		$(if $(filter $(DISABLE_TQDM),1),--disable-tqdm,) \
		--epochs $(EPOCHS) \
		--early-stopping-patience $(ES_PATIENCE) --early-stopping-min-delta $(ES_MIN_DELTA) \
		$(if $(filter $(SKIP_IF_EXISTS),1),--skip-if-exists,)

# Uses the trained model dir from markup_base.yaml
# Adjust --max-length if needed

# Eval assumes model saved to $(OUTDIR)
# (markup_base.yaml default is artifacts/markup_run)

eval:
	$(PY) scripts/eval_markup.py --model-dir $(OUTDIR) \
		--val-jsonl data/pages_val.jsonl \
		--test-jsonl data/pages_test.jsonl \
		--max-length $(MAXLEN) \
		$(if $(filter $(MARKUP_TRUNCATE),1),--max-html-chars $(MARKUP_MAX_HTML_CHARS),--max-html-chars -1)
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended predictions with --tag _full"; \
		$(PY) scripts/eval_markup.py --model-dir $(OUTDIR) \
			--val-jsonl data/pages_val_full.jsonl \
			--test-jsonl data/pages_test_full.jsonl \
			--max-length $(MAXLEN) \
			$(if $(filter $(MARKUP_TRUNCATE),1),--max-html-chars $(MARKUP_MAX_HTML_CHARS),--max-html-chars -1) \
			--tag _full; \
	fi

.PHONY: train-js eval-js
train-js:
	$(PY) scripts/train_js_codet5p.py --train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/js_codet5p --model-name Salesforce/codet5p-220m --max-length 512 --batch-size $(if $(filter $(SMOKE),1),2,4) --num-epochs $(if $(filter $(SMOKE),1),1,1) --lr 3e-5 \
		$(if $(filter $(DISABLE_TQDM),1),--disable-tqdm,) \
		--early-stopping-patience $(ES_PATIENCE) --early-stopping-min-delta $(ES_MIN_DELTA)

eval-js:
	$(PY) scripts/eval_js_codet5p.py --model-dir artifacts/js_codet5p --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --max-length 512
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended predictions with --tag _full"; \
		$(PY) scripts/eval_js_codet5p.py --model-dir artifacts/js_codet5p --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --max-length 512 --tag _full; \
	fi

# Lightweight heads (URL/JS CharCNN and DOM GCN)
.PHONY: train-url-head train-js-head train-dom-gcn
train-url-head:
	$(PY) scripts/train_url_head.py --train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/url_head --batch-size $(if $(filter $(SMOKE),1),32,64) --epochs $(if $(filter $(SMOKE),1),1,3) --lr 1e-3 --weight-decay 1e-4

train-js-head:
	$(PY) scripts/train_js_head.py --train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/js_charcnn --batch-size $(if $(filter $(SMOKE),1),16,32) --epochs $(if $(filter $(SMOKE),1),1,3) --lr 1e-3 --weight-decay 1e-4 --num-workers $(JS_HEAD_NUM_WORKERS) --resume $(if $(filter $(DISABLE_TQDM),1),--disable-tqdm,) $(if $(filter $(AUGMENT_JS),1),--raw-field js_augmented,)

train-dom-gcn:
	$(PY) scripts/train_dom_gcn.py --train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/dom_gcn --batch-size $(if $(filter $(SMOKE),1),8,16) --epochs $(if $(filter $(SMOKE),1),1,3) --lr 2e-3 --weight-decay 1e-4

# Phase 3: Text and Cheap-feature heads
.PHONY: train-text-head train-cheap-mlp
train-text-head:
	$(PY) scripts/train_text_head.py --train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/text_head --batch-size $(if $(filter $(SMOKE),1),16,32) --epochs $(if $(filter $(SMOKE),1),1,3) --lr 1e-3 --weight-decay 1e-4 --max-len $(if $(filter $(SMOKE),1),512,1024)

HIDDEN_CHEAP ?= 128
train-cheap-mlp:
	$(PY) scripts/train_cheap_mlp.py --train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/cheap_mlp --batch-size $(if $(filter $(SMOKE),1),64,128) --epochs $(if $(filter $(SMOKE),1),2,5) --lr 2e-3 --weight-decay 1e-4 --hidden $(HIDDEN_CHEAP)

.PHONY: fuse
fuse:
	$(PY) scripts/fuse_heads.py --dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --dom-light-dir artifacts/dom_gcn --text-dir artifacts/text_head --cheap-mlp-dir artifacts/cheap_mlp --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --out-dir artifacts/fusion --method logistic --use-cheap-features
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended fusion with --head-tag _full --tag _full"; \
		$(PY) scripts/fuse_heads.py --dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --dom-light-dir artifacts/dom_gcn --text-dir artifacts/text_head --cheap-mlp-dir artifacts/cheap_mlp --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --out-dir artifacts/fusion --method logistic --use-cheap-features --head-tag _full --tag _full; \
	fi

# Comprehensive coverage-max fusion (imputation) and unified export
.PHONY: fuse-all-coverage
fuse-all-coverage:
	$(PY) scripts/fuse_heads.py \
		--dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --dom-light-dir artifacts/dom_gcn --text-dir artifacts/text_head --cheap-mlp-dir artifacts/cheap_mlp \
		--val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl \
		--out-dir artifacts/fusion_all --method logistic --alignment-strategy coverage_max --min-heads 2 --use-cheap-features --export-unified-json
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended fusion_all with --head-tag _full --tag _full"; \
		$(PY) scripts/fuse_heads.py \
			--dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --dom-light-dir artifacts/dom_gcn --text-dir artifacts/text_head --cheap-mlp-dir artifacts/cheap_mlp \
			--val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl \
			--out-dir artifacts/fusion_all --method logistic --alignment-strategy coverage_max --min-heads 2 --use-cheap-features --export-unified-json --head-tag _full --tag _full; \
	fi

# Second-level meta-fusion including prior first-level fusion head
.PHONY: fuse-meta
fuse-meta: fuse-all-coverage
	$(PY) scripts/fuse_heads.py \
		--dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --dom-light-dir artifacts/dom_gcn --text-dir artifacts/text_head --cheap-mlp-dir artifacts/cheap_mlp \
		--val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl \
		--out-dir artifacts/fusion_meta --method logistic --alignment-strategy coverage_max --min-heads 2 --use-cheap-features --include-fusion-head --fusion-dir artifacts/fusion_all --export-unified-json
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended fusion_meta with --head-tag _full --tag _full"; \
		$(PY) scripts/fuse_heads.py \
			--dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --dom-light-dir artifacts/dom_gcn --text-dir artifacts/text_head --cheap-mlp-dir artifacts/cheap_mlp \
			--val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl \
			--out-dir artifacts/fusion_meta --method logistic --alignment-strategy coverage_max --min-heads 2 --use-cheap-features --include-fusion-head --fusion-dir artifacts/fusion_all --export-unified-json --head-tag _full --tag _full; \
	fi

# Meta weight search (simplex) producing fusion_meta/weights.json & preds
.PHONY: meta-fuse-search
meta-fuse-search:
	$(PY) scripts/meta_fuse_heads.py \
		--val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl \
		--out-dir artifacts/fusion_meta \
		--dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --dom-light-dir artifacts/dom_gcn --text-dir artifacts/text_head --cheap-mlp-dir artifacts/cheap_mlp \
		--alignment-strategy inner_join --strategy random --random-samples 4000 --dirichlet-alpha 1.0
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Running meta-fusion weight search on extended splits"; \
		$(PY) scripts/meta_fuse_heads.py \
			--val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl \
			--out-dir artifacts/fusion_meta \
			--dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --dom-light-dir artifacts/dom_gcn --text-dir artifacts/text_head --cheap-mlp-dir artifacts/cheap_mlp \
			--alignment-strategy inner_join --strategy random --random-samples 4000 --dirichlet-alpha 1.0 --head-tag _full --tag _full; \
	fi
# Cross-attention fusion (XFusion)
.PHONY: train-xfusion eval-xfusion
train-xfusion:
	$(PY) scripts/train_fusion_xattn_fixed.py \
		--train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl \
		--out-dir artifacts/fusion_xattn \
		$(if $(filter $(XF_NO_URL),1),--no-url,) \
		$(if $(filter $(XF_NO_JS),1),--no-js,) \
		$(if $(filter $(XF_NO_TEXT),1),--no-text,) \
		$(if $(filter $(XF_NO_DOM),1),--no-dom,) \
		$(if $(filter $(XF_NO_CHEAP),1),--no-cheap,) \
		$(if $(XF_JS_RAW_FIELD),--js-raw-field $(XF_JS_RAW_FIELD),) \
		$(if $(filter $(XF_NO_JS_CANON),1),--no-js-canonicalize,) \
		$(if $(filter $(XF_HTML_CANON),1),--html-canonicalize,) \
		$(if $(XF_HTML_FIELD),--html-field $(XF_HTML_FIELD),)

.PHONY: train-xfusion-diag
train-xfusion-diag:
	$(PY) scripts/train_fusion_xattn_fixed.py \
		--train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl \
		--out-dir artifacts/fusion_xattn \
		--record-diagnostics --diag-interval 100 \
		$(if $(filter $(XF_NO_URL),1),--no-url,) \
		$(if $(filter $(XF_NO_JS),1),--no-js,) \
		$(if $(filter $(XF_NO_TEXT),1),--no-text,) \
		$(if $(filter $(XF_NO_DOM),1),--no-dom,) \
		$(if $(filter $(XF_NO_CHEAP),1),--no-cheap,) \
		$(if $(XF_JS_RAW_FIELD),--js-raw-field $(XF_JS_RAW_FIELD),) \
		$(if $(filter $(XF_NO_JS_CANON),1),--no-js-canonicalize,) \
		$(if $(filter $(XF_HTML_CANON),1),--html-canonicalize,) \
		$(if $(XF_HTML_FIELD),--html-field $(XF_HTML_FIELD),)

eval-xfusion: train-xfusion
	@echo "[MAKE] XFusion trained and preds/calibration written to artifacts/fusion_xattn"
# Run cascade after fusion
.PHONY: cascade
cascade:
	$(PY) scripts/cascade.py --url-dir artifacts/url_head --cheap-dir artifacts/cheap_mlp --fusion-dir $(if $(filter $(USE_XFUSION),1),artifacts/fusion_xattn,artifacts/fusion) --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --out-dir artifacts/cascade --target-precision 0.99 --target-benign-precision 0.995

.PHONY: report
report:
	$(PY) scripts/report_eval.py --model-dir $(OUTDIR) --train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --max-length $(MAXLEN) --device cuda --eval-batch 4
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] NOTE: Extended dataset available - consider updating report script to use _full files"; \
	fi

.PHONY: report-xai
report-xai:
	$(PY) scripts/report_eval.py --model-dir $(OUTDIR) \
		--train-jsonl data/pages_train.jsonl \
		--val-jsonl data/pages_val.jsonl \
		--test-jsonl data/pages_test.jsonl \
		--max-length $(MAXLEN) \
		--device cuda --eval-batch 4 \
		--lime --shap --num-expl 1 \
		--xai-device $(XAI_DEVICE) --xai-max-chars 1500 --xai-num-samples 150 --xai-background 3
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] NOTE: Extended dataset available - consider updating report script to use _full files"; \
	fi

# Evaluate calibrated lightweight heads and generate preds jsonl
.PHONY: eval-url-head eval-js-head eval-dom-gcn
eval-url-head:
	$(PY) scripts/eval_light_heads.py --head url --model-dir artifacts/url_head --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --batch-size 128
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended predictions with --tag _full"; \
		$(PY) scripts/eval_light_heads.py --head url --model-dir artifacts/url_head --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --batch-size 128 --tag _full; \
	fi

eval-js-head:
	$(PY) scripts/eval_light_heads.py --head js --model-dir artifacts/js_charcnn --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --batch-size 64
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended predictions with --tag _full"; \
		$(PY) scripts/eval_light_heads.py --head js --model-dir artifacts/js_charcnn --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --batch-size 64 --tag _full; \
	fi

# Phase 6 (optional): JS augmentation and augmented head
.PHONY: phase6
phase6: augment-js train-js-head-aug eval-js-head-aug eval-dom-gcn eval-text-head eval-cheap-mlp report

# Phase 6: JS augmentation and optional retraining using augmented field
.PHONY: augment-js train-js-head-aug eval-js-head-aug
augment-js:
	$(PY) scripts/augment_js.py --in-jsonl data/pages_train.jsonl --out-jsonl data/pages_train_aug.jsonl --prob-hex $(if $(filter $(SMOKE),1),0.01,0.05) --prob-split $(if $(filter $(SMOKE),1),0.01,0.05) --seed $(if $(SEED),$(SEED),42)

train-js-head-aug:
	$(PY) scripts/train_js_head.py --train-jsonl data/pages_train_aug.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/js_charcnn_aug --batch-size $(if $(filter $(SMOKE),1),16,32) --epochs $(if $(filter $(SMOKE),1),1,3) --lr 1e-3 --weight-decay 1e-4 --raw-field js_augmented --num-workers $(JS_HEAD_NUM_WORKERS) --resume $(if $(filter $(DISABLE_TQDM),1),--disable-tqdm,)

eval-js-head-aug:
	$(PY) scripts/eval_light_heads.py --head js --model-dir artifacts/js_charcnn_aug --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --batch-size 64
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended predictions with --tag _full"; \
		$(PY) scripts/eval_light_heads.py --head js --model-dir artifacts/js_charcnn_aug --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --batch-size 64 --tag _full; \
	fi

eval-dom-gcn:
	$(PY) scripts/eval_light_heads.py --head dom --model-dir artifacts/dom_gcn --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --batch-size 32
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended predictions with --tag _full"; \
		$(PY) scripts/eval_light_heads.py --head dom --model-dir artifacts/dom_gcn --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --batch-size 32 --tag _full; \
	fi

.PHONY: eval-text-head eval-cheap-mlp
eval-text-head:
	$(PY) scripts/eval_light_heads.py --head text --model-dir artifacts/text_head --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --batch-size 64
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended predictions with --tag _full"; \
		$(PY) scripts/eval_light_heads.py --head text --model-dir artifacts/text_head --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --batch-size 64 --tag _full; \
	fi

eval-cheap-mlp:
	$(PY) scripts/eval_light_heads.py --head cheap --model-dir artifacts/cheap_mlp --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --batch-size 256
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended predictions with --tag _full"; \
		$(PY) scripts/eval_light_heads.py --head cheap --model-dir artifacts/cheap_mlp --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --batch-size 256 --tag _full; \
	fi

# Plot PR curves comparing individual heads and fused output
.PHONY: plot-heads
plot-heads:
	$(PY) scripts/plot_eval_heads.py --url-dir artifacts/url_head --js-dir artifacts/js_charcnn --dom-dir artifacts/dom_gcn --fusion-dir artifacts/fusion --split test --out artifacts/fusion/heads_pr_curves.png

# End-to-end: fetch feeds, unify to seed, optional crawl, then splits/train/eval
ifeq ($(CRAWL),false)
all e2e: feeds unify crawl-verify ensure-crawled-count auto-backfill splits slice auto-backfill phase4 $(if $(filter $(USE_XFUSION),1),train-xfusion,)
else ifeq ($(CRAWL),0)
all e2e: feeds unify crawl-verify ensure-crawled-count auto-backfill splits slice auto-backfill phase4 $(if $(filter $(USE_XFUSION),1),train-xfusion,)
else ifeq ($(CRAWL),no)
all e2e: feeds unify crawl-verify ensure-crawled-count auto-backfill splits slice auto-backfill phase4 $(if $(filter $(USE_XFUSION),1),train-xfusion,)
else
all e2e: feeds unify crawl ensure-crawled-count auto-backfill splits slice auto-backfill phase4 $(if $(filter $(AUGMENT_JS),1),phase6,) $(if $(filter $(USE_XFUSION),1),train-xfusion,)
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
resume: splits slice train eval train-js-head eval-js-head train-dom-gcn eval-dom-gcn train-text-head eval-text-head train-cheap-mlp eval-cheap-mlp

clean:
	rm -f data/pages.jsonl data/pages_train.jsonl data/pages_val.jsonl data/pages_test.jsonl data/splits.json
	rm -rf artifacts/markup_run
	rm -rf artifacts/url_head artifacts/js_charcnn artifacts/dom_gcn artifacts/text_head artifacts/cheap_mlp artifacts/fusion

# Quick smoke crawl of first 5 URLs to verify new fields populate
.PHONY: smoke
smoke:
	@head -n 1 data/seed.csv > data/seed_smoke.csv
	@head -n 6 data/seed.csv | tail -n +2 >> data/seed_smoke.csv
	$(PY) scripts/crawl_playwright.py --input-csv data/seed_smoke.csv --out-jsonl data/pages_smoke.jsonl --concurrency 2 --timeout-s $(CRAWL_TIMEOUT) --block-assets --no-external-js --retries 1 --tls-timeout $(TLS_TIMEOUT) --dns-timeout $(DNS_TIMEOUT) $(if $(filter $(MOBILE_PROFILE),true),--mobile-profile,) $(if $(filter $(GPU),true),--gpu,)
	@echo "--- sample record with lightweight fields ---" && head -n 1 data/pages_smoke.jsonl | $(PY) -c 'import sys,json;obj=json.loads(sys.stdin.read());keep=["url_final","redirect_hops","url_len","dns_created_days_ago","cert_age_days","hdr_csp","req_unique_etld1","form_pw_count","js_entropy","fp_canvas","phash64","favicon_dhash","bitb_like_modal","qr_flag","cloak_delta_domlen"];print(json.dumps({k:obj.get(k) for k in keep if k in obj}, indent=2))'

.PHONY: backfill
backfill:
	$(PY) scripts/backfill_fields.py --inputs data/pages.jsonl data/pages_train.jsonl data/pages_val.jsonl data/pages_test.jsonl --overwrite --network --tls-timeout $(TLS_TIMEOUT) --dns-timeout $(DNS_TIMEOUT)

.PHONY: auto-backfill
auto-backfill:
	@echo "[MAKE] Auto backfill (network=$(BACKFILL_NETWORK)) [parallel workers=$(BACKFILL_WORKERS) batch=$(BACKFILL_BATCH) visuals=$(if $(filter $(BACKFILL_DISABLE_VISUALS),1),off,on) drop_raw=$(BACKFILL_DROP_RAW) max_img=$(BACKFILL_MAX_IMAGE_BYTES)B]"
	$(PY) scripts/auto_backfill.py --inputs data/pages.jsonl data/pages_train.jsonl data/pages_val.jsonl data/pages_test.jsonl --overwrite $(if $(filter $(BACKFILL_NETWORK),1),--network,) --tls-timeout $(TLS_TIMEOUT) --dns-timeout $(DNS_TIMEOUT) $(if $(BACKFILL_WORKERS),--workers $(BACKFILL_WORKERS),) $(if $(BACKFILL_BATCH),--batch-lines $(BACKFILL_BATCH),) $(if $(filter $(BACKFILL_DISABLE_VISUALS),1),--disable-visuals,) $(if $(BACKFILL_MAX_IMAGE_BYTES),--max-image-bytes $(BACKFILL_MAX_IMAGE_BYTES),) $(if $(filter $(BACKFILL_DROP_RAW),1),--drop-raw,)

# Phase 4 end-to-end: train/eval all heads, fuse, plot, report
.PHONY: phase4
phase4: train eval report train-js eval-js train-url-head eval-url-head train-js-head eval-js-head train-dom-gcn eval-dom-gcn train-text-head eval-text-head train-cheap-mlp eval-cheap-mlp fuse cascade report report-xai plot-heads

# Phase 8: Cross-attention + robustness (HTML/JS canonicalization)
.PHONY: phase8
phase8: phase4 $(if $(filter $(AUGMENT_JS),1),phase6,) train-xfusion cascade report report-xai
