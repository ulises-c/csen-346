#!/usr/bin/env bash
# Local llama.cpp server for the SocratDataset translation job.
# Run via: make start-local-tl-server
set -euo pipefail

~/Github/llama.cpp/build/bin/llama-server \
    -m ~/models/Qwen3.5-9B/Qwen3.5-9B-UD-Q4_K_XL.gguf \
    --host 0.0.0.0 --port 8000 \
    -ngl 99 \
    -c 32768 \
    -t 10 -tb 20 \
    --no-think
