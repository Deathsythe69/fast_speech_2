#!/usr/bin/env bash
# run_inference.sh <lang> <checkpoint>
# Runs inference.py, plot_variance.py, sentence_stats.py, eval_val.py in order
# and writes Results/<Lang>/inference_manifest.json

set -e

LANG_ARG=${1:-english}
CKPT_ARG=${2:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_EXEC=".venv/Scripts/python.exe"
else
    PYTHON_EXEC="python"
fi

if [ -z "$CKPT_ARG" ]; then
    "$PYTHON_EXEC" tools/run_pipeline.py --lang "$LANG_ARG"
else
    "$PYTHON_EXEC" tools/run_pipeline.py --lang "$LANG_ARG" --checkpoint "$CKPT_ARG"
fi

