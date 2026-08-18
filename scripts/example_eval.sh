#!/usr/bin/env bash
# Minimal example: generate and score belief-confirmation responses for one
# (model, verb, prompt template). Run from the repository root after setting up
# .env and the data files (see README.md and data/README.md).
set -euo pipefail

MODEL="${MODEL:-meta-llama/llama-3.1-8b-instruct}"
TEMPLATE="${TEMPLATE:-original}"
VERB="${VERB:-believe}"

python -m src.eval.main --mode generate \
  --model "$MODEL" --prompt_template "$TEMPLATE" --verb "$VERB"

python -m src.eval.main --mode evaluate \
  --model "$MODEL" --prompt_template "$TEMPLATE" --verb "$VERB"
