#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-${ROOT_DIR}/checkpoints/bert-base-uncased}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO_ID="${REPO_ID:-google-bert/bert-base-uncased}"

mkdir -p "${TARGET_DIR}"

export HF_ENDPOINT
export HF_HUB_DISABLE_TELEMETRY=1

python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${REPO_ID}",
    local_dir=r"${TARGET_DIR}",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "vocab.txt",
        "model.safetensors",
    ],
)
print("Downloaded bert-base-uncased safetensors to: ${TARGET_DIR}")
PY

if [ -f "${TARGET_DIR}/model.safetensors" ] && [ -f "${TARGET_DIR}/pytorch_model.bin" ]; then
  rm -f "${TARGET_DIR}/pytorch_model.bin"
fi
