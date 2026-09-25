#!/usr/bin/env bash
# scripts/setup_cloud_env.sh
#
# Build the pinned Python 3.11 environment on Kaggle or Colab, isolated from
# the platform's default stack (Kaggle 2026-09: Python 3.12 / torch 2.10 /
# numpy 2.0 — none match our pins). Verified on Kaggle GPU T4 x2, 2026-09-25:
# all pins exact, torch 2.3.1+cu121 runs on driver 580, pytest 352 passed.
#
# Usage (from the repo root):  bash scripts/setup_cloud_env.sh
# Afterwards run code with:    $VENV/bin/python ...  (PYTHONPATH set as below)
set -euo pipefail

if [ -d /kaggle/working ]; then BASE=/kaggle/working
elif [ -d /content ]; then BASE=/content
else BASE="$(pwd)/.."; fi
VENV="${VENV:-$BASE/venv311}"
PY="$VENV/bin/python"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

pip install -q uv
uv python install 3.11
[ -x "$PY" ] || uv venv -q --python 3.11 "$VENV"

# torch first from the CUDA 12.1 index (PyPI would also give cu121 for 2.3.1,
# but pinning the index makes the CUDA build explicit).
uv pip install -q --python "$PY" --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1
# everything else straight from requirements.txt (git+ts2vec line skipped:
# not pip-installable at the pinned commit; we use the vendored copy).
grep -v '^git+' "$REPO/requirements.txt" > /tmp/req_nogit.txt
uv pip install -q --python "$PY" -r /tmp/req_nogit.txt

export PYTHONPATH="$REPO:$REPO/third_party_reference/ts2vec"
"$PY" "$REPO/scripts/check_env.py"
echo "VENV=$VENV"
