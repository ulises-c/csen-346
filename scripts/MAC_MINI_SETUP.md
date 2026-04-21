# Mac Mini Ollama Setup

This doc covers running the KELE consultant (Qwen3.5-9B) via **Ollama** on an
M4 Mac Mini (16 GB), accessible to a host PC over LAN.

The teacher (SocratTeachLLM) still runs on WAVE or another GPU node — only the
consultant is offloaded to the Mac Mini.

---

## Hardware

| Device | Role | Model |
|---|---|---|
| M4 Mac Mini 16 GB | Consultant server | Qwen3.5-9B via Ollama |
| Host PC | Eval runner + teacher | SocratTeachLLM via vLLM / WAVE |

Observed throughput on M4 Mac Mini:
- **~13 tokens/s** (from quick test, 1972-token response)
- Runs entirely in Apple Neural Engine / Metal — no CUDA required

---

## One-time setup (Mac Mini)

Run the setup script once after installing Ollama:

```bash
bash scripts/mac_mini_setup.sh
```

This:
1. Verifies Ollama is installed and `qwen3.5:9b` is pulled
2. Sets `OLLAMA_HOST=0.0.0.0` via `launchctl setenv` so Ollama binds to all
   interfaces (not just localhost)
3. Restarts Ollama to apply the new binding
4. Prints the Mac Mini's LAN IP and the env vars to set on the host PC

### What OLLAMA_HOST=0.0.0.0 does

By default Ollama binds to `127.0.0.1:11434`, rejecting any connection that
doesn't originate on the same machine. Setting `OLLAMA_HOST=0.0.0.0` makes it
bind to all network interfaces, allowing the host PC on the same LAN to connect.

### macOS firewall

If the host PC can't reach port 11434 after setup:

1. **System Settings → Network → Firewall → Options**
2. Confirm Ollama is listed and set to "Allow incoming connections"
3. If the firewall is on and Ollama isn't listed, click **+** and add
   `/usr/local/bin/ollama` (or wherever `which ollama` points)

---

## Starting the consultant (each session)

If Ollama.app is already running with `OLLAMA_HOST=0.0.0.0`, nothing extra is
needed — just confirm connectivity:

```bash
./scripts/serve_consultant_ollama.sh
```

This verifies the server is up, confirms the model is loaded, warms it up, and
prints the endpoint for the host PC.

If Ollama isn't running (e.g. after a reboot before opening Ollama.app):

```bash
# Option A: open Ollama.app (picks up OLLAMA_HOST from launchctl)
open -a Ollama
# then run the script to confirm

# Option B: start the CLI server directly
OLLAMA_HOST=0.0.0.0 ollama serve
```

---

## Running the eval from the host PC

Set these env vars on the host PC before running the eval:

```bash
export CONSULTANT_BASE_URL=http://<mac-ip>:11434/v1
export CONSULTANT_MODEL_NAME=qwen3.5:9b
export CONSULTANT_API_KEY=ollama          # Ollama ignores the key; required by OpenAI client
```

Then run:

```bash
./scripts/run_eval.sh local
```

### Finding the Mac Mini's IP

```bash
# On the Mac Mini:
ipconfig getifaddr en0

# Or check System Settings → Network → Wi-Fi / Ethernet → Details → IP Address
```

For a stable IP, assign a DHCP reservation in your router using the Mac Mini's
MAC address — this prevents the IP changing between sessions.

---

## Ollama's OpenAI-compatible API

Ollama exposes `/v1/chat/completions` and `/v1/models` at port 11434, matching
the OpenAI SDK interface. The eval code calls the consultant identically
regardless of whether it's talking to Ollama or vLLM — only the base URL and
model name change.

```bash
# Test from host PC:
curl http://<mac-ip>:11434/v1/models

# Quick generation test:
curl http://<mac-ip>:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:9b",
    "messages": [{"role": "user", "content": "say hi"}]
  }'
```

---

## Performance notes

| Setup | Consultant | Est. eval time |
|---|---|---|
| WAVE V100 × 2 | Qwen3.5-9B via vLLM (GPU 1) | ~34 hrs |
| Mac Mini M4 16 GB | Qwen3.5-9B via Ollama | ~TBD (~4-bit quantized) |

The Ollama `qwen3.5:9b` model is Q4_K_M quantized (6.6 GB) — fits entirely in
unified memory. Throughput (~13 tok/s observed) will be slower than the V100's
~25 tok/s unquantized, but no cluster queue time and no 48h time limit.

> Run a full eval to fill in the actual time estimate and add it to the table above.

---

## Keeping OLLAMA_HOST persistent across reboots

`launchctl setenv` only survives until the next reboot. To make it permanent:

```bash
# Create a LaunchAgent that sets the env var before Ollama.app auto-starts
mkdir -p ~/Library/LaunchAgents
cat > ~/Library/LaunchAgents/com.ollama.env.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ollama.env</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/launchctl</string>
        <string>setenv</string>
        <string>OLLAMA_HOST</string>
        <string>0.0.0.0</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.ollama.env.plist
```

After the next reboot, `OLLAMA_HOST=0.0.0.0` will be set before Ollama.app
starts, so it will bind to all interfaces automatically.
