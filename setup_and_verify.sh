#!/usr/bin/env bash
# setup_and_verify.sh
#
# One-shot setup + baseline verification script for continuing this
# project in a full environment (Claude Code, local machine, Colab,
# etc.) after migrating from the network-less Claude.ai sandbox where
# M0-M6 were originally built.
#
# What this does:
#   1. Installs dependencies from requirements.txt
#   2. Runs the full test suite as a baseline sanity check
#   3. Reports a clear pass/fail summary
#
# What this does NOT do:
#   - Does not run M1 (data acquisition) — that requires you to
#     actually want to download real Binance data, which takes time
#     and bandwidth. Run `python scripts/run_m1_acquisition.py` yourself
#     when ready.
#   - Does not modify any source code, even if tests fail.
#
# Usage:
#   bash setup_and_verify.sh

set -uo pipefail  # NOTE: deliberately not using -e, so we can report
                   # a full summary even if a step fails partway.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=================================================================="
echo "  Market State Discovery — Setup & Baseline Verification"
echo "  (Migration from Claude.ai sandbox -> full environment)"
echo "=================================================================="
echo ""

STEP_1_OK=0
STEP_2_OK=0
STEP_3_OK=0

# --- Step 1: Install dependencies ---
echo "--- Step 1/3: Installing dependencies from requirements.txt ---"
if pip install -r requirements.txt; then
    echo -e "${GREEN}[OK]${NC} Dependencies installed."
    STEP_1_OK=1
else
    echo -e "${RED}[FAIL]${NC} pip install failed. Check the error above."
    echo "Common causes: no network access, TS2Vec commit hash unreachable"
    echo "(see configs/base.yaml ts2vec: block for the pinned commit and"
    echo "fallback plan), or a genuine version conflict."
fi
echo ""

# --- Step 2: Verify key imports work ---
echo "--- Step 2/3: Verifying key imports ---"
python3 -c "
import sys
failures = []
try:
    import torch
    print(f'[OK] torch {torch.__version__}')
except ImportError as e:
    failures.append(f'torch: {e}')

try:
    import pandas
    print(f'[OK] pandas {pandas.__version__}')
except ImportError as e:
    failures.append(f'pandas: {e}')

try:
    import numpy
    print(f'[OK] numpy {numpy.__version__}')
except ImportError as e:
    failures.append(f'numpy: {e}')

try:
    import pytest
    print(f'[OK] pytest {pytest.__version__}')
except ImportError as e:
    failures.append(f'pytest: {e}')

try:
    from ts2vec import TS2Vec
    print('[OK] ts2vec (pinned commit) importable')
except ImportError as e:
    failures.append(f'ts2vec: {e}')

try:
    import ccxt
    print(f'[OK] ccxt {ccxt.__version__}')
except ImportError as e:
    failures.append(f'ccxt: {e}')

try:
    import hdbscan
    print(f'[OK] hdbscan {hdbscan.__version__}')
except ImportError as e:
    failures.append(f'hdbscan: {e}')

if failures:
    print()
    print('[FAIL] Missing imports:')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
else:
    print()
    print('[OK] All key dependencies importable.')
    sys.exit(0)
"
if [ $? -eq 0 ]; then
    STEP_2_OK=1
fi
echo ""

# --- Step 3: Run the full test suite (M0-M6) ---
echo "--- Step 3/3: Running test suite (M0-M6 baseline) ---"
echo "This is the first time this test suite has run with real pytest —"
echo "it was previously only verified via manual script execution in a"
echo "sandbox without pytest installed. Read output carefully."
echo ""
if pytest tests/ -v --tb=short; then
    echo -e "${GREEN}[OK]${NC} All tests passed."
    STEP_3_OK=1
else
    echo -e "${YELLOW}[ATTENTION]${NC} Some tests failed or errored."
    echo "Do NOT immediately 'fix' the tests. First determine whether this is:"
    echo "  (a) A genuine bug in src/ that was missed during manual verification, or"
    echo "  (b) An environment difference (package version, OS) requiring a"
    echo "      careful, documented fix — see docs/CHECKPOINT_LATEST.md for"
    echo "      two precedents where this exact situation occurred (pandas"
    echo "      3.x datetime64 unit changes)."
    echo "Document your finding in docs/CHECKPOINT_LATEST.md before proceeding."
fi
echo ""

# --- Summary ---
echo "=================================================================="
echo "  SUMMARY"
echo "=================================================================="
[ $STEP_1_OK -eq 1 ] && echo -e "  Dependencies installed:  ${GREEN}OK${NC}" || echo -e "  Dependencies installed:  ${RED}FAILED${NC}"
[ $STEP_2_OK -eq 1 ] && echo -e "  Key imports verified:    ${GREEN}OK${NC}" || echo -e "  Key imports verified:    ${RED}FAILED${NC}"
[ $STEP_3_OK -eq 1 ] && echo -e "  Test suite (M0-M6):      ${GREEN}ALL PASSED${NC}" || echo -e "  Test suite (M0-M6):      ${YELLOW}SEE OUTPUT ABOVE${NC}"
echo ""
echo "Next steps:"
echo "  - If everything above is OK: read MIGRATION_TO_CLAUDE_CODE.md,"
echo "    then docs/IMP-01_v1.3.md, and begin M7 (TS2Vec Wrapper)."
echo "  - If M1 data hasn't been downloaded yet, run:"
echo "      python scripts/run_m1_acquisition.py --config configs/base.yaml"
echo "    followed by M2-M6 in sequence (see docs/IMP-01_v1.3.md Coding Order)."
echo "=================================================================="
