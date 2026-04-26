# scripts/

Shell scripts for environment setup and process management. Not for Python tooling.

| Script | Purpose |
|--------|---------|
| `run_eval.sh` | Launch a batch evaluation run |
| `serve_both.sh` | Start teacher + consultant vLLM servers together |
| `serve_dual_gpu.sh` | Serve teacher (GPU 0) and consultant (GPU 1) on dual-GPU hosts |
| `serve_socratteachllm.sh` | Serve SocratTeachLLM teacher only |
| `serve_consultant.sh` | Serve consultant model via vLLM |
| `serve_consultant_llamacpp.sh` | Serve consultant model via llama.cpp (Mac Mini) |
| `serve_teacher_online.sh` | Serve teacher using an online API endpoint |
| `serve_gemma4.sh` | Serve Gemma 4 as the teacher |
| `post_eval_shutdown.sh` | Shut down vLLM servers after a run completes |
| `l40s_setup.sh` | One-time environment setup for L40S hosts |
| `mac_mini_setup.sh` | One-time environment setup for Mac Mini |
| `wave_setup.sh` | One-time environment setup for Wave cluster |

Setup docs (`.md`) for each hardware target live alongside their setup scripts.
Python utilities belong in `src/project/` instead.
