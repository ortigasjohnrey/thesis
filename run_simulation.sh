#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -f ".venv/bin/activate" ]; then
  echo "Virtual environment not found. Run setup_env.sh first."
  exit 1
fi
source .venv/bin/activate
python scripts/run_simulation.py
