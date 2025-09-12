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
DIAG_INTERVAL ?= 100
REPORT_OUT ?= artifacts/report_e2e
INCLUDE_XFUSION ?= 1

# Optional toggle to include heavy CodeT5p JS head in unified report
ENABLE_CODET5P ?= 0
XF_DIAG_DIR ?= artifacts/fusion_xattn/diagnostics
XF_BATCH_SIZE ?= 8
XF_D_MODEL ?= 96
XF_N_LAYERS ?= 2
XF_N_HEADS ?= 4
XF_FF_MULT ?= 4
XF_TOKEN_EMBED ?= 48
XF_AMP ?= 1
XF_GRAD_CHECKPOINT ?= 0
XF_SMOKE_LIMIT ?= 512
XF_ATTN_DROPOUT ?=
XF_FF_DROPOUT ?=
XF_REVERSIBLE ?= 0

# Versioned splits configuration
WRITE_V2 ?= 0
SPLITS_VERSION_TAG ?= v2_default
SPLITS_VERSION_FILE ?= data/splits_v2.json

# Heads directories list for reports (pruned: removing dom_gcn & cheap_mlp as requested)
HEADS_DIRS = artifacts/url_head artifacts/text_head artifacts/js_charcnn
HEADS_DIRS += $(if $(filter $(ENABLE_CODET5P),1),artifacts/js_codet5p,)

# Optional per-user overrides (create local.mk without committing secrets)
-include local.mk

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
		if [ -f data/pages_phish_only.jsonl ]; then \
			echo "[MAKE] Ingesting phish-only additions into master dataset"; \
			$(PY) scripts/ingest_phish_only.py --dataset data/pages.jsonl --phish-only data/pages_phish_only.jsonl || exit 3; \
		fi; \
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
		$(if $(filter $(BALANCE_SPLITS),1),--balance-splits,) \
		$(if $(filter $(WRITE_V2),1),--write-v2 --version-tag $(SPLITS_VERSION_TAG),)
	@if [ -f $(SPLITS_VERSION_FILE) ]; then echo "[MAKE] Versioned splits present: $(SPLITS_VERSION_FILE)"; fi

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

# Lightweight heads (URL/JS CharCNN only; dom_gcn removed)
.PHONY: train-url-head train-js-head
train-url-head:
	$(PY) scripts/train_url_head.py --train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/url_head --batch-size $(if $(filter $(SMOKE),1),32,64) --epochs $(if $(filter $(SMOKE),1),1,3) --lr 1e-3 --weight-decay 1e-4

train-js-head:
	$(PY) scripts/train_js_head.py --train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/js_charcnn --batch-size $(if $(filter $(SMOKE),1),16,32) --epochs $(if $(filter $(SMOKE),1),1,3) --lr 1e-3 --weight-decay 1e-4 --num-workers $(JS_HEAD_NUM_WORKERS) --resume $(if $(filter $(DISABLE_TQDM),1),--disable-tqdm,) $(if $(filter $(AUGMENT_JS),1),--raw-field js_augmented,)

# Phase 3: Text head only (cheap MLP removed)
.PHONY: train-text-head
train-text-head:
	$(PY) scripts/train_text_head.py --train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/text_head --batch-size $(if $(filter $(SMOKE),1),16,32) --epochs $(if $(filter $(SMOKE),1),1,3) --lr 1e-3 --weight-decay 1e-4 --max-len $(if $(filter $(SMOKE),1),512,1024)

.PHONY: fuse
fuse:
		$(PY) scripts/fuse_heads.py --dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --text-dir artifacts/text_head --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --out-dir artifacts/fusion --method logistic
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
			echo "[MAKE] Generating extended fusion with --tag _full (head tag now auto-inferred)"; \
			$(PY) scripts/fuse_heads.py --dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --text-dir artifacts/text_head --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --out-dir artifacts/fusion --method logistic --tag _full; \
	fi

# Comprehensive coverage-max fusion (imputation) and unified export
.PHONY: fuse-all-coverage
fuse-all-coverage:
	$(PY) scripts/fuse_heads.py \
		--dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --text-dir artifacts/text_head \
		--val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl \
		--out-dir artifacts/fusion_all --method logistic --alignment-strategy coverage_max --min-heads 1 --export-unified-json
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended fusion_all with --head-tag _full --tag _full"; \
		$(PY) scripts/fuse_heads.py \
			--dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --text-dir artifacts/text_head \
			--val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl \
			--out-dir artifacts/fusion_all --method logistic --alignment-strategy coverage_max --min-heads 1 --export-unified-json --head-tag _full --tag _full; \
	fi

# Second-level meta-fusion including prior first-level fusion head
.PHONY: fuse-meta
fuse-meta: fuse-all-coverage
	$(PY) scripts/fuse_heads.py \
		--dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --text-dir artifacts/text_head \
		--val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl \
		--out-dir artifacts/fusion_meta --method logistic --alignment-strategy coverage_max --min-heads 1 --include-fusion-head --fusion-dir artifacts/fusion_all --export-unified-json
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended fusion_meta with --head-tag _full --tag _full"; \
		$(PY) scripts/fuse_heads.py \
			--dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --text-dir artifacts/text_head \
			--val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl \
			--out-dir artifacts/fusion_meta --method logistic --alignment-strategy coverage_max --min-heads 1 --include-fusion-head --fusion-dir artifacts/fusion_all --export-unified-json --head-tag _full --tag _full; \
	fi

# Meta weight search (simplex) producing fusion_meta/weights.json & preds
.PHONY: meta-fuse-search
# Phase targets (Phases 2-11)
.PHONY: phase2-cluster phase3-augment phase5-stack-cv phase6-cascade phase7-loss phase8-refiner phase9-eval phase10-gate phase11-kpi

phase2-cluster:
	$(PY) scripts/cluster_false_positives.py
	$(PY) scripts/cluster_false_positives_enhanced.py

phase3-augment:
	$(PY) scripts/prepare_hard_negatives.py
	$(PY) scripts/augment_with_hard_negatives.py
	$(PY) scripts/schedule_retrain_with_hard_negatives.py

# Phase 3b: Retrain heads using augmented data (reads class weights from plan)
.PHONY: phase3-retrain
phase3-retrain: phase3-augment
	@echo "[MAKE] Hard negative retrain starting (reading plan manifest)"
	@PLAN=artifacts/diagnostics/hard_negative_retrain_plan.json; \
	if [ ! -f $$PLAN ]; then echo "[MAKE][ERROR] Missing retrain plan $$PLAN"; exit 3; fi; \
	POSW=$$(python - <<- 'PY'
	import json; import sys
	p=json.load(open('artifacts/diagnostics/hard_negative_retrain_plan.json'))
	cw=p.get('heads',[{}])[0].get('class_weights',{})
	pos=cw.get('positive',1.0); neg=cw.get('negative',1.0)
	# BCEWithLogitsLoss pos_weight multiplies positive class loss; negatives weight=1.
	# Use ratio so relative weighting matches plan suggestions.
	pw = (pos/neg) if neg else pos
	print(f"{pw:.6f}")
	PY
	); \
	echo "[MAKE] Using pos_weight=$$POSW from plan"; \
	$(PY) scripts/train_url_head.py --train-jsonl data/pages_train_augmented.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/url_head --batch-size $(if $(filter $(SMOKE),1),32,64) --epochs $(if $(filter $(SMOKE),1),1,3) --lr 1e-3 --weight-decay 1e-4 --pos-weight $$POSW; \
	$(PY) scripts/train_js_head.py --train-jsonl data/pages_train_augmented.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/js_charcnn --batch-size $(if $(filter $(SMOKE),1),16,32) --epochs $(if $(filter $(SMOKE),1),1,3) --lr 1e-3 --weight-decay 1e-4 --num-workers $(JS_HEAD_NUM_WORKERS) --pos-weight $$POSW $(if $(filter $(DISABLE_TQDM),1),--disable-tqdm,); \
	$(PY) scripts/train_text_head.py --train-jsonl data/pages_train_augmented.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/text_head --batch-size $(if $(filter $(SMOKE),1),16,32) --epochs $(if $(filter $(SMOKE),1),1,3) --lr 1e-3 --weight-decay 1e-4 --max-len $(if $(filter $(SMOKE),1),512,1024) --pos-weight $$POSW; \
	$(PY) scripts/train_js_codet5p.py --train-jsonl data/pages_train_augmented.jsonl --val-jsonl data/pages_val.jsonl --output-dir artifacts/js_codet5p --model-name Salesforce/codet5p-220m --max-length 512 --batch-size $(if $(filter $(SMOKE),1),2,4) --num-epochs $(if $(filter $(SMOKE),1),0.5,1.0) --lr 3e-5 $(if $(filter $(DISABLE_TQDM),1),--disable-tqdm,); \
	echo "[MAKE] (Optional) Provide your CodeT5p retrain command if desired"

.PHONY: eval-heads-postretrain
eval-heads-postretrain:
	$(PY) scripts/eval_light_heads.py --head url --model-dir artifacts/url_head --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --batch-size 128
	$(PY) scripts/eval_light_heads.py --head js --model-dir artifacts/js_charcnn --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --batch-size 64
	$(PY) scripts/eval_light_heads.py --head text --model-dir artifacts/text_head --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --batch-size 64
	$(PY) scripts/eval_js_codet5p.py --model-dir artifacts/js_codet5p --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --max-length 512

phase5-stack-cv:
	$(PY) scripts/refit_fusion_calibrated_ids_cv.py
	$(PY) scripts/refit_meta_fusion_cv.py --include-fusion-prob

phase6-cascade:
	$(PY) scripts/cascade_optimize_v2.py
	$(PY) scripts/cascade_band_search.py

phase7-loss:
	$(PY) scripts/fusion_loss_experiments.py

.PHONY: phase7-adopt
phase7-adopt:
	$(PY) scripts/refit_fusion_calibrated_ids.py --adopt-best-loss

phase8-refiner:
	$(PY) scripts/train_confusion_refiner.py --eval-on-test

phase9-eval:
	$(PY) scripts/evaluate_enhanced.py
	$(PY) scripts/compare_stacking.py

phase10-gate:
	$(PY) scripts/ci_check_anomalies.py --split val
	$(PY) scripts/ci_gate.py $(if $(CREATE_BASELINE),--create-baseline,)
	$(PY) scripts/ci_gate_smoke.py || true

phase11-kpi:
	$(PY) scripts/build_kpi_dashboard.py

meta-fuse-search:
	$(PY) scripts/meta_fuse_heads.py \
		--val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl \
		--out-dir artifacts/fusion_meta \
		--dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --text-dir artifacts/text_head \
		--alignment-strategy inner_join --strategy random --random-samples 4000 --dirichlet-alpha 1.0
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Running meta-fusion weight search on extended splits"; \
		$(PY) scripts/meta_fuse_heads.py \
			--val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl \
			--out-dir artifacts/fusion_meta \
			--dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --text-dir artifacts/text_head \
			--alignment-strategy inner_join --strategy random --random-samples 4000 --dirichlet-alpha 1.0 --head-tag _full --tag _full; \
	fi
# Cross-attention fusion (XFusion)
.PHONY: train-xfusion eval-xfusion
train-xfusion:
	$(PY) scripts/train_fusion_xattn_fixed.py \
		--train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl \
		--out-dir artifacts/fusion_xattn \
		--d-model $(XF_D_MODEL) --n-heads $(XF_N_HEADS) --n-layers $(XF_N_LAYERS) --ff-mult $(XF_FF_MULT) --token-embed-dim $(XF_TOKEN_EMBED) $(if $(filter $(XF_AMP),1),--amp,) $(if $(filter $(XF_GRAD_CHECKPOINT),1),--grad-checkpoint,) \
		$(if $(XF_ATTN_DROPOUT),--attn-dropout $(XF_ATTN_DROPOUT),) $(if $(XF_FF_DROPOUT),--ff-dropout $(XF_FF_DROPOUT),) $(if $(filter $(XF_REVERSIBLE),1),--reversible,) \
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
		--batch-size $(XF_BATCH_SIZE) \
		--d-model $(XF_D_MODEL) --n-heads $(XF_N_HEADS) --n-layers $(XF_N_LAYERS) --ff-mult $(XF_FF_MULT) --token-embed-dim $(XF_TOKEN_EMBED) $(if $(filter $(XF_AMP),1),--amp,) $(if $(filter $(XF_GRAD_CHECKPOINT),1),--grad-checkpoint,) \
		$(if $(XF_ATTN_DROPOUT),--attn-dropout $(XF_ATTN_DROPOUT),) $(if $(XF_FF_DROPOUT),--ff-dropout $(XF_FF_DROPOUT),) $(if $(filter $(XF_REVERSIBLE),1),--reversible,) \
		--record-diagnostics --diag-interval $(DIAG_INTERVAL) --diag-dir $(XF_DIAG_DIR) \
		$(if $(filter $(XF_NO_URL),1),--no-url,) \
		$(if $(filter $(XF_NO_JS),1),--no-js,) \
		$(if $(filter $(XF_NO_TEXT),1),--no-text,) \
		$(if $(filter $(XF_NO_DOM),1),--no-dom,) \
		$(if $(filter $(XF_NO_CHEAP),1),--no-cheap,) \
		$(if $(XF_JS_RAW_FIELD),--js-raw-field $(XF_JS_RAW_FIELD),) \
		$(if $(filter $(XF_NO_JS_CANON),1),--no-js-canonicalize,) \
		$(if $(filter $(XF_HTML_CANON),1),--html-canonicalize,) \
		$(if $(XF_HTML_FIELD),--html-field $(XF_HTML_FIELD),)

.PHONY: xfusion-smoke
xfusion-smoke:
	$(PY) scripts/train_fusion_xattn_fixed.py \
		--train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl \
		--out-dir artifacts/fusion_xattn_smoke \
		--batch-size $(if $(filter $(XF_BATCH_SIZE),),$(XF_BATCH_SIZE),4) \
		--epochs 1 --limit-train $(XF_SMOKE_LIMIT) --limit-val $(XF_SMOKE_LIMIT) --limit-test $(XF_SMOKE_LIMIT) \
		--d-model $(XF_D_MODEL) --n-heads $(XF_N_HEADS) --n-layers 1 --ff-mult 2 --token-embed-dim $(XF_TOKEN_EMBED) $(if $(filter $(XF_AMP),1),--amp,) $(if $(filter $(XF_GRAD_CHECKPOINT),1),--grad-checkpoint,) \
		$(if $(XF_ATTN_DROPOUT),--attn-dropout $(XF_ATTN_DROPOUT),) $(if $(XF_FF_DROPOUT),--ff-dropout $(XF_FF_DROPOUT),) $(if $(filter $(XF_REVERSIBLE),1),--reversible,) \
		--record-diagnostics --diag-interval 50 --diag-dir artifacts/fusion_xattn_smoke/diagnostics \
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
# Cascade removed (cheap/dom heads removed)

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

# Extended heads prediction (generate _full for all active heads if extended splits exist)
.PHONY: eval-extended-heads
eval-extended-heads:
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Evaluating extended heads (_full)"; \
		$(PY) scripts/eval_markup.py --model-dir artifacts/markup_run --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --max-length $(MAXLEN) --max-html-chars -1 --tag _full; \
		$(PY) scripts/eval_js_codet5p.py --model-dir artifacts/js_codet5p --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --max-length 512 --tag _full; \
		$(PY) scripts/eval_light_heads.py --head url --model-dir artifacts/url_head --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --batch-size 128 --tag _full || echo "[WARN] url head extended eval failed"; \
		$(PY) scripts/eval_light_heads.py --head text --model-dir artifacts/text_head --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --batch-size 64 --tag _full || echo "[WARN] text head extended eval failed"; \
		$(PY) scripts/eval_light_heads.py --head js --model-dir artifacts/js_charcnn --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --batch-size 64 --tag _full || echo "[WARN] js_charcnn base extended eval failed"; \
		$(PY) scripts/eval_light_heads.py --head js --model-dir artifacts/js_charcnn_aug --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --batch-size 64 --tag _full || echo "[WARN] js_charcnn_aug extended eval failed"; \
	else \
		echo "[MAKE] Extended split files not found; skipping eval-extended-heads"; \
	fi

# Fusion variants regeneration (strict, coverage_max, soft, meta)
.PHONY: fuse-variants
fuse-variants: fuse fuse-all-coverage fuse-meta
	$(PY) scripts/fuse_heads_coverage_max.py --min-heads 2 || true
	$(PY) scripts/fuse_heads_soft_impute.py || true
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended coverage_max fusion_covmax (_full)"; \
		$(PY) scripts/fuse_heads.py --alignment-strategy coverage_max --min-heads 2 --dom-dir artifacts/markup_run --js-dir artifacts/js_codet5p --url-dir artifacts/url_head --text-dir artifacts/text_head --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --out-dir artifacts/fusion_covmax --head-tag _full --tag _full --normalize-dom-model || true; \
		echo "[MAKE] Generating extended soft fusion (_full)"; \
		$(PY) scripts/fuse_heads_soft_impute.py --full || true; \
	fi

# Generate strict baseline including fusion variants
.PHONY: baseline-strict
baseline-strict:
	$(PY) scripts/export_predictions.py --mode strict --model-allowlist dom js_code fused fused_covmax fused_soft meta_fused url text js_charcnn_base js_charcnn_aug --out-dir artifacts/baseline_strict
	$(PY) scripts/audit_coverage_extended.py --baseline-dir artifacts/baseline_strict --model-allowlist dom js_code fused fused_covmax fused_soft meta_fused url text js_charcnn_base js_charcnn_aug

# Extended strict baseline (includes *_full variants when available in a separate directory)
.PHONY: baseline-extended
baseline-extended: eval-extended-heads
	$(PY) scripts/export_predictions.py --mode strict --model-allowlist dom js_code fused fused_covmax fused_soft meta_fused url text js_charcnn_base js_charcnn_aug --out-dir artifacts/baseline_strict_full --separate-full
	$(PY) scripts/audit_coverage_extended.py --baseline-dir artifacts/baseline_strict_full --model-allowlist dom js_code fused fused_covmax fused_soft meta_fused url text js_charcnn_base js_charcnn_aug

# Calibration comparison report across fusion variants
.PHONY: calibrate-fusion-report
calibrate-fusion-report:
	$(PY) scripts/calibrate_fusion_variants.py --baseline-dir artifacts/baseline_strict --out artifacts/diagnostics/fusion_calibration_report.md || echo "[WARN] calibration report script missing"

# Unified target to refresh everything after crawl/ingest
.PHONY: full-refresh
full-refresh: ensure-crawled-count eval eval-js eval-url-head train-js-head train-text-head fuse-variants baseline-strict calibrate-fusion-report


# Evaluate calibrated lightweight heads and generate preds jsonl
.PHONY: eval-url-head eval-js-head
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
phase6: augment-js train-js-head-aug eval-js-head-aug eval-text-head report

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


.PHONY: eval-text-head
eval-text-head:
	$(PY) scripts/eval_light_heads.py --head text --model-dir artifacts/text_head --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl --batch-size 64
	@if [ -f data/pages_val_full.jsonl ] && [ -f data/pages_test_full.jsonl ]; then \
		echo "[MAKE] Generating extended predictions with --tag _full"; \
		$(PY) scripts/eval_light_heads.py --head text --model-dir artifacts/text_head --val-jsonl data/pages_val_full.jsonl --test-jsonl data/pages_test_full.jsonl --batch-size 64 --tag _full; \
	fi


# Plot PR curves comparing individual heads and fused output
.PHONY: plot-heads
plot-heads:
	$(PY) scripts/plot_eval_heads.py --url-dir artifacts/url_head --js-dir artifacts/js_charcnn --fusion-dir artifacts/fusion --split test --out artifacts/fusion/heads_pr_curves.png

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
resume: splits slice train eval train-js-head eval-js-head train-text-head eval-text-head

clean:
	rm -f data/pages.jsonl data/pages_train.jsonl data/pages_val.jsonl data/pages_test.jsonl data/splits.json
	rm -rf artifacts/markup_run
	rm -rf artifacts/url_head artifacts/js_charcnn artifacts/text_head artifacts/fusion

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

# ---- ID Repair and Balanced Subset Utilities ----
.PHONY: repair-ids
repair-ids:
	$(PY) scripts/repair_ids.py --in data/pages.jsonl $(if $(wildcard data/pages_train.jsonl),data/pages_train.jsonl,) $(if $(wildcard data/pages_val.jsonl),data/pages_val.jsonl,) $(if $(wildcard data/pages_test.jsonl),data/pages_test.jsonl,)

.PHONY: make-balanced
make-balanced: repair-ids
	@mkdir -p data
	$(PY) scripts/make_balanced_subset.py --in data/pages.jsonl --out data/pages_balanced.jsonl --n $(if $(N_PER_CLASS),$(N_PER_CLASS),6000) --seed $(if $(SEED),$(SEED),42) --shuffle
	@echo "[MAKE] Balanced dataset at data/pages_balanced.jsonl"

.PHONY: splits-balanced
splits-balanced: make-balanced
	$(MAKE) splits DATASET=data/pages_balanced.jsonl POSR_TRAIN=0.5 POSR_VAL=0.5 POSR_TEST=0.5 BALANCE_SPLITS=1 SEED=$(if $(SEED),$(SEED),42) MIN_TOTAL_TEST=$(if $(MIN_TOTAL_TEST),$(MIN_TOTAL_TEST),1000)
	$(MAKE) slice DATASET=data/pages_balanced.jsonl

.PHONY: pipeline-balanced
pipeline-balanced: splits-balanced
	$(MAKE) run-all-custom-nosplits DATASET=data/pages_balanced.jsonl $(if $(ADOPT_BEST_LOSS),ADOPT_BEST_LOSS=$(ADOPT_BEST_LOSS),) $(if $(CREATE_BASELINE),CREATE_BASELINE=$(CREATE_BASELINE),) $(if $(HEAD_COSTS_JSON),HEAD_COSTS_JSON=$(HEAD_COSTS_JSON),) $(if $(COST_SCALE),COST_SCALE=$(COST_SCALE),)
	# Auto-generate fresh full report after balanced pipeline completes
	$(MAKE) report-full

# Phase 4 end-to-end: train/eval all heads, fuse, plot, report
.PHONY: phase4
phase4: train eval report train-js eval-js train-url-head eval-url-head train-js-head eval-js-head train-text-head eval-text-head fuse report report-xai plot-heads

# Phase 8: Cross-attention + robustness (HTML/JS canonicalization)
.PHONY: phase8
phase8: phase4 $(if $(filter $(AUGMENT_JS),1),phase6,) train-xfusion report report-xai

# --- New unified orchestration targets (attention entropy, meta fusion, comprehensive report) ---
.PHONY: train-heads eval-heads fuse-coverage meta-fuse-all xfusion-diag report-full full-e2e

train-heads: train-url-head train-js-head train-text-head

eval-heads: eval-url-head eval-js-head eval-text-head

# Coverage-max fusion (reuses existing target fuse-all-coverage)
fuse-coverage: fuse-all-coverage

# Meta fusion stacked on coverage fusion
meta-fuse-all: fuse-meta meta-fuse-search

# Cross-attention with diagnostics (attention entropy trend + alerts)
xfusion-diag: train-xfusion-diag

# Comprehensive HTML report including heads, meta fusion, cascade, and XFusion diagnostics
report-full:
	@mkdir -p $(REPORT_OUT)
	$(PY) scripts/report_eval.py \
		--out-dir $(REPORT_OUT) \
		--val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl \
		--fusion-dir artifacts/fusion_covmax \
		$(if $(filter $(INCLUDE_XFUSION),1),--xfusion-diag artifacts/fusion_xattn/diagnostics/diagnostics.json,) \
		$(if $(filter $(INCLUDE_XFUSION),1),--xfusion-dir artifacts/fusion_xattn,) \
		--heads-dirs artifacts/url_head artifacts/text_head artifacts/js_charcnn $(if $(filter $(ENABLE_CODET5P),1),artifacts/js_codet5p,) \
		--meta-fusion-dir artifacts/fusion_meta || (echo "[MAKE][ERROR] report failed" && exit 4)
	@echo "[MAKE] Report generated at $(REPORT_OUT)/index.html"

# Extended XAI + versioned splits aware report
.PHONY: report-extended-xai
report-extended-xai:
	@mkdir -p $(REPORT_OUT)
	$(PY) scripts/report_eval.py \
		--out-dir $(REPORT_OUT) \
		--model-dir artifacts/markup_run \
		--train-jsonl data/pages_train.jsonl --val-jsonl data/pages_val.jsonl --test-jsonl data/pages_test.jsonl \
		--full-jsonl data/pages.jsonl \
		--meta-fusion-dir artifacts/fusion_meta \
		--fusion-dir artifacts/fusion \
		--splits-version $(SPLITS_VERSION_FILE) \
		$(if $(filter $(USE_XFUSION),1),--xfusion-diag $(XF_DIAG_DIR)/diagnostics.json,) \
		$(if $(filter $(USE_XFUSION),1),--xfusion-dir artifacts/fusion_xattn,) \
		--heads-dirs $(HEADS_DIRS) \
		--device $(XAI_DEVICE) --eval-batch 4 \
		--lime --shap --num-expl 1 \
		--xai-device $(XAI_DEVICE) --xai-max-chars 1500 --xai-num-samples 150 --xai-background 3 || (echo "[MAKE][ERROR] extended report failed" && exit 4)
	@echo "[MAKE] Extended report at $(REPORT_OUT)/index.html"

# Full pipeline after data prepared & splits (does NOT crawl). Adjust sequence as needed.
full-e2e: splits slice train-heads eval-heads fuse-coverage meta-fuse-all xfusion-diag report-full
	@echo "[MAKE] Full E2E (with attention entropy diagnostics) complete."

# Convenience alias
.PHONY: all-full
all-full: full-e2e

# Ultra pipeline mirroring manual multi-command sequence (heads + augmentation + diagnostics fusion + coverage/meta + meta search + extended XAI report)
.PHONY: e2e-ultra
e2e-ultra: ensure-crawled-count extend-splits-if-needed splits slice train-url-head train-js-head train-text-head \
	eval-url-head eval-js-head eval-text-head \
	$(if $(filter $(AUGMENT_JS),1),augment-js train-js-head-aug eval-js-head-aug,) \
	fuse-all-coverage fuse-meta meta-fuse-search train-xfusion-diag report-extended-xai
	@echo "[MAKE] e2e-ultra pipeline complete (diag interval=$(DIAG_INTERVAL), splits tag=$(SPLITS_VERSION_TAG))"

# ---- Custom unified pipeline integrating new phases ----
.PHONY: cascade-cost-sweep
cascade-cost-sweep:
	@echo "[MAKE] Cascade cost sweep with default costs"
	$(PY) scripts/cascade_optimize_v2.py --split val $(if $(HEAD_COSTS_JSON),--head-costs-json $(HEAD_COSTS_JSON),) $(if $(COST_SCALE),--cost-scale $(COST_SCALE),)
	@for s in 0.5 0.75 1.0 1.25 1.5 ; do \
		echo "[MAKE] Sensitivity cost_scale=$$s"; \
		$(PY) scripts/cascade_optimize_v2.py --split val $(if $(HEAD_COSTS_JSON),--head-costs-json $(HEAD_COSTS_JSON),) --cost-scale $$s || true; \
	done

.PHONY: hard-neg-cycle
hard-neg-cycle: phase2-cluster phase3-augment phase3-retrain eval-heads-postretrain phase5-stack-cv phase7-loss phase7-adopt phase6-cascade phase8-refiner phase9-eval phase10-gate phase11-kpi
	@echo "[MAKE] Hard-negative cycle complete"

# Master pipeline: assumes crawl has been run externally if desired
.PHONY: run-all-custom
run-all-custom: ensure-crawled-count extend-splits-if-needed splits slice \
	train-heads eval-heads fuse-variants fuse-coverage meta-fuse-all \
	phase5-stack-cv phase7-loss $(if $(filter $(ADOPT_BEST_LOSS),1),phase7-adopt,) \
	phase6-cascade cascade-cost-sweep phase8-refiner phase9-eval phase10-gate phase11-kpi
	@echo "[MAKE] run-all-custom complete (ADOPT_BEST_LOSS=$(ADOPT_BEST_LOSS))"

# Convenience: run both standard flow then optional hard-negative cycle
.PHONY: run-all-with-hardnegs
run-all-with-hardnegs: run-all-custom $(if $(filter $(DO_HARD_NEG_CYCLE),1),hard-neg-cycle,)
	@echo "[MAKE] run-all-with-hardnegs complete (cycle=$(DO_HARD_NEG_CYCLE))"

# Friendly alias: single-command pipeline with optional variables
.PHONY: pipeline
pipeline: run-all-custom
	@echo "[MAKE] pipeline finished (alias of run-all-custom)"

# Variant: master pipeline without splits/extend to respect pre-made splits
.PHONY: run-all-custom-nosplits
run-all-custom-nosplits: \
	train-heads eval-heads fuse-variants fuse-coverage meta-fuse-all \
	phase5-stack-cv phase7-loss $(if $(filter $(ADOPT_BEST_LOSS),1),phase7-adopt,) \
	phase6-cascade cascade-cost-sweep phase8-refiner phase9-eval phase10-gate phase11-kpi
	@echo "[MAKE] run-all-custom-nosplits complete (ADOPT_BEST_LOSS=$(ADOPT_BEST_LOSS))"
