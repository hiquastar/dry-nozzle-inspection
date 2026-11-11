#!/usr/bin/env bash
# Chapter 5 deployment protocol for the final model, both seeds.
#   ./run.sh                 # creates .venv on the first run
# Extra arguments are passed to benchmark.py (e.g. --threads 4).
# Results: results/pi5_s42.json, results/pi5_s123.json
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    python3 -m venv .venv
    . .venv/bin/activate
    pip install --upgrade --quiet pip
    pip install --quiet -r requirements.txt
else
    . .venv/bin/activate
fi

python3 benchmark.py --seed 42 "$@"
echo "cooling down 60 s before seed 123 ..."
sleep 60
python3 benchmark.py --seed 123 "$@"
