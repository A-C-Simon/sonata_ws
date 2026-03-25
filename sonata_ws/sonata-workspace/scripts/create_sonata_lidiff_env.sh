#!/usr/bin/env bash
# Create or update conda env sonata_lidiff (see environment.yml).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx sonata_lidiff; then
  echo "Updating existing env sonata_lidiff..."
  conda env update -f environment.yml --prune
else
  echo "Creating env sonata_lidiff..."
  conda env create -f environment.yml
fi

echo ""
echo "Next (editable install of this repo):"
echo "  conda activate sonata_lidiff"
echo "  cd \"$ROOT\""
echo "  pip install -e ."
