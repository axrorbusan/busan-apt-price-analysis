#!/usr/bin/env bash
# 전체 분석 실행 (원본 데이터가 data/raw 에 있다고 가정)
#   데이터 수집부터 하려면: python3 scripts/00_collect.py trades → 01 → 00 geocode → 00 poi
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output data/processed
python3 scripts/01_clean.py
python3 scripts/02_features.py
python3 scripts/03_model.py
python3 scripts/04_figures.py
echo "완료 → output/"
