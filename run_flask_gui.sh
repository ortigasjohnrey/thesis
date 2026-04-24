#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "Virtual environment not found. Run: bash setup_env.sh"
  exit 1
fi
source .venv/bin/activate
python app.py
