#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "CUDA sync is only intended for Linux GPU hosts."
  exit 1
fi

uv sync --python 3.11 --extra cuda --extra data
