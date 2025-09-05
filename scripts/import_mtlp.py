#!/usr/bin/env python
"""
This script is intentionally disabled.

External dataset support (e.g., MTLP) has been removed. The project now uses
only feed-based data and crawling. If you reached this script, please migrate
to the feeds-only workflow via the Makefile targets: feeds -> unify -> crawl -> splits -> slice -> train -> eval.
"""

raise SystemExit(
	"import_mtlp.py has been removed: external dataset support is no longer included."
)
