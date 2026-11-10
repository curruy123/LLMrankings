#!/usr/bin/env bash
# Runs the full pipeline in order and saves each script's console log.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs/figures
python 01_fit_overall.py        | tee outputs/log_01_overall.txt
python 02_fit_categories.py     | tee outputs/log_02_categories.txt
python 03_compare_rankings.py   | tee outputs/log_03_compare.txt
python 04_evaluate_predictions.py | tee outputs/log_04_predictions.txt
python 04b_sensitivity.py       | tee outputs/log_04b_sensitivity.txt
echo "All steps finished."
