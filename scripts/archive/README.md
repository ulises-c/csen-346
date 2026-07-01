# scripts/archive/

Retired scripts kept for provenance. **Not wired into the Makefile** and not part of
any current workflow — they document how past runs were driven and let you reconstruct
a run's exact settings. New work should use the entry points in [`../README.md`](../README.md).

## Retired per-model eval shells

The 20 `eval_*.sh` orchestrators, superseded by the parameterized
[`../eval_llamacpp.sh`](../eval_llamacpp.sh) (single-server llama.cpp serve+eval+compare)
and, for the current run, by the Makefile→`kele` path.

- **Single-server (now `eval_llamacpp.sh` config rows):** `eval_qwen27b.sh`,
  `eval_qwen27b_think.sh`, `eval_qwen35b_a3b.sh`, `eval_qwen35b_a3b_n50.sh`,
  `eval_qwen35b_a3b_n50_think.sh`, `eval_qwopus35b_a3b.sh`, `eval_gemma4_31b.sh`,
  `eval_gemma4_26b_a4b.sh`
- **BERT few-shot orchestrators:** `eval_bert_a3b_fewshot10_full.sh`,
  `eval_bert_gemma_fewshot10_full.sh`, `eval_bert_glm49b_chat_fewshot10_full.sh`,
  `eval_bert_qwen27b_fewshot10_full.sh`, `eval_bert_qwen27b_nothink_fewshot10_full.sh`,
  `eval_bert_socratteachllm_fewshot10_full.sh`, `eval_bert_claude_fewshot10.sh`
- **AMD Linux + Mac Mini (2-machine):** `eval_amd_mac.sh`, `eval_amd_2mac.sh`,
  `eval_amd_2mac_mlx.sh`
- **Claude / prompt-tournament:** `eval_claude_consultant_socratteachllm.sh`,
  `eval_prompt_tournament.sh`

## Dated one-off chains / backtests

`autonomous_chain_2026_05_16.sh`, `overnight_qwen27b_chain.sh`,
`run_all_fusion_smokes.sh`, `post_chain_backtest.sh`, `post_phase3_backtest.sh`.
