# Mac Mini llama.cpp Setup

This doc covers running the KELE consultant (Qwen3.5-9B) via **llama.cpp** on an
M4 Mac Mini (16 GB), accessible to a host PC over LAN.

The teacher (SocratTeachLLM) still runs on the AMD host — only the consultant is
offloaded to the Mac Mini.

---

## Hardware

| Device | Role | Stack |
|---|---|---|
| M4 Mac Mini 16 GB | Consultant server | Qwen3.5-9B via llama.cpp (Metal) |
| AMD R9700 host | Eval runner + teacher | SocratTeachLLM via vLLM |

---

## One-time setup (Mac Mini)

### 1. Install llama.cpp

```bash
brew install llama.cpp
```

Verify:

```bash
llama-server --version
```

### 2. Download the model

Get a Q4_K_M GGUF from Hugging Face (fits in 16 GB unified memory):

```
https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF
```

Set `CONSULTANT_MODEL_PATH` in `configs/local-mac-m4.env` to the absolute path of the
downloaded `.gguf` file.

### 3. Run the setup check

```bash
set -a && source configs/local-mac-m4.env && set +a
bash scripts/mac_mini_setup.sh
```

This verifies `llama-server` is in PATH, the model file exists, and prints the Mac Mini's
LAN address + the env vars to set on the host PC.

### 4. macOS firewall

If the host PC can't reach port `8080`:

1. **System Settings → Network → Firewall → Options**
2. Confirm that incoming connections on port `8080` are allowed (or add `llama-server`)

---

## Starting the consultant (each session)

Run this on the Mac Mini before starting the eval on the host PC:

```bash
set -a && source configs/local-mac-m4.env && set +a
./scripts/serve_consultant_llamacpp.sh
```

The script:
1. Starts `llama-server` with full Metal GPU offload (`-ngl 99`) if not already running
2. Warms up the model so the first eval call isn't slow
3. Inhibits sleep with `caffeinate` for the lifetime of the server
4. Prints the endpoint and the env vars to set on the host PC

If `llama-server` is already running on port `8080` the script just confirms connectivity
and attaches `caffeinate` — no duplicate process.

---

## Running the eval from the host PC

```bash
./scripts/eval_amd_mac.sh              # full run (resumes if interrupted)
./scripts/eval_amd_mac.sh --limit 5   # smoke test
./scripts/eval_amd_mac.sh --limit 5 --new  # fresh run, archives previous results
```

`eval_amd_mac.sh` automatically:
1. Checks the Mac Mini consultant is reachable
2. Starts the AMD teacher server in the background
3. Waits for the teacher to be ready
4. Runs the eval via `run_eval.sh local-mac-m4`

---

## Addressing the Mac Mini

**Preferred — `.local` hostname (no config needed):**

```bash
# Confirm the hostname on the Mac Mini:
scutil --get LocalHostName   # → Ulisess-Mac-mini
# Reachable as: Ulisess-Mac-mini.local
```

macOS broadcasts this via Bonjour. Works on any device on the same LAN without touching
DNS or your router.

**Fallback — IP address:**

```bash
# On the Mac Mini:
ipconfig getifaddr en0
```

For a stable IP, assign a DHCP reservation in your router using the Mac Mini's MAC address.

---

## API compatibility

llama.cpp's `llama-server` exposes `/v1/chat/completions` and `/v1/models` at port `8080`,
matching the OpenAI SDK interface. The eval code calls the consultant identically regardless
of backend — only the base URL and model name change.

```bash
# Test from host PC:
curl http://Ulisess-Mac-mini.local:8080/v1/models

# Quick generation test:
curl http://Ulisess-Mac-mini.local:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"consultant","messages":[{"role":"user","content":"say hi"}],"max_tokens":5}'
```

---

## Configuration reference (`configs/local-mac-m4.env`)

| Variable | Value | Notes |
|---|---|---|
| `CONSULTANT_BASE_URL` | `http://Ulisess-Mac-mini.local:8080/v1` | LAN endpoint |
| `CONSULTANT_MODEL_NAME` | `<basename of .gguf>` | Reported to the OpenAI client |
| `CONSULTANT_API_KEY` | `no-key` | llama.cpp ignores auth; required by OpenAI SDK |
| `CONSULTANT_MODEL_PATH` | `/path/to/model.gguf` | Absolute path on the Mac Mini |
| `CONSULTANT_GPU_LAYERS` | `99` | Full Metal offload for M4 |
| `CONSULTANT_NUM_CTX` | `16384` | Context window |

---

## Performance notes

| Setup | Consultant | Est. eval time |
|---|---|---|
| WAVE V100 × 2 | Qwen3.5-9B via vLLM | ~34 hrs |
| Mac Mini M4 16 GB | Qwen3.5-9B via llama.cpp (Q4_K_M) | TBD |

> Run a full eval to fill in the actual time estimate and update the table above.
