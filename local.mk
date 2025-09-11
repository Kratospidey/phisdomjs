# Local overrides (not to be committed) matching requested environment
# Automatically included by main Makefile via: -include local.mk

# --- Requested variable values ---
CRAWL=false
GPU=true
CRAWL_TIMEOUT=8.0
CRAWL_CONCURRENCY=12
CRAWL_RETRIES=4
SEED_SIZE=30000
PHISH_RATIO=0.5
MIN_CRAWLED_URLS=14750
EPOCHS=1
SKIP_IF_EXISTS=1
XAI_DEVICE=cuda
BACKFILL_WORKERS=3
BACKFILL_BATCH=200
BACKFILL_NETWORK=0
BACKFILL_DROP_RAW=0
AUGMENT_JS=1
USE_XFUSION=0
XF_NO_TEXT=0
XF_HTML_CANON=1
# Optional: include CodeT5p head by default? Uncomment if desired
ENABLE_CODET5P=1
# Diagnostics interval (kept from previous defaults; adjust if needed)
DIAG_INTERVAL=100
WRITE_V2=1
SPLITS_VERSION_TAG=v2_run1
SPLITS_VERSION_FILE=data/splits_v2.json

# --- Composite sequence target replicating manual chain ---
# Order preserved from user command:
# ensure-crawled-count extend-splits-if-needed phase4 phase6 \
# train-xfusion-diag fuse-all-coverage fuse-meta meta-fuse-search report-xai

CUSTOM_SEQUENCE_TARGETS=ensure-crawled-count extend-splits-if-needed \
  phase4 phase6 \
  $(if $(filter $(USE_XFUSION),1),train-xfusion-diag,) \
  fuse-all-coverage fuse-meta meta-fuse-search \
  report-extended-xai

# Note: train-xfusion-diag is now conditional on USE_XFUSION=1

.PHONY: custom-sequence
custom-sequence: $(CUSTOM_SEQUENCE_TARGETS)
	@echo "[local.mk] Completed custom sequence: $(CUSTOM_SEQUENCE_TARGETS)"

# Convenience alias
.PHONY: run-all-custom
run-all-custom: custom-sequence
	@echo "[local.mk] run-all-custom finished"
