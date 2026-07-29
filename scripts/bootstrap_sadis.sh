#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$ROOT_DIR/vendor"
SADIS_DIR="$VENDOR_DIR/SADis"

mkdir -p "$VENDOR_DIR"
if [[ -d "$SADIS_DIR/.git" ]]; then
  echo "SADis already exists at $SADIS_DIR"
  git -C "$SADIS_DIR" pull --ff-only
else
  git clone https://github.com/deepffff/SADis.git "$SADIS_DIR"
fi

echo "SADis is ready at $SADIS_DIR"
echo "Place IP-Adapter weights under models/ and sdxl_models/ as described in README.md."
