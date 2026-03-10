#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

apt-get update
apt-get install -y git python3-venv tmux

cd "$ROOT_DIR"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools wheel
pip install -e ".[dev,research]"

echo "bootstrap complete: $ROOT_DIR"
