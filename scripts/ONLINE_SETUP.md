# Online Setup

This repo can already run locally on your RTX 3070. If you want to run it
"online" from the same machine, the simplest architecture is:

1. Keep the model server running on your own GPU box
2. Bind it to `127.0.0.1`
3. Put a tunnel or reverse proxy in front of it
4. Require an API key before forwarding requests

That gives you a reachable HTTPS endpoint without moving the model to a cloud GPU.

## Recommended path: tunnel your 3070 box

The repo now includes [`scripts/serve_teacher_online.sh`](scripts/serve_teacher_online.sh),
which starts the teacher server in a safer online-serving mode:

```bash
TEACHER_LOCAL_PATH=~/hf_models/SocratTeachLLM \
TEACHER_SERVER_API_KEY=replace-this-with-a-long-random-secret \
./scripts/serve_teacher_online.sh
```

Behavior:

- binds to `127.0.0.1:8001` by default
- requires either `Authorization: Bearer <key>` or `x-api-key: <key>`
- exposes a health endpoint at `/healthz`

### Example with Cloudflare Tunnel

If you already use Cloudflare:

```bash
cloudflared tunnel --url http://127.0.0.1:8001
```

That gives you a public HTTPS URL that forwards to your local GPU server.

### Example with ngrok

```bash
ngrok http http://127.0.0.1:8001
```

Use the generated HTTPS URL as your public endpoint.

### Test the public endpoint

```bash
curl https://your-public-url.example/v1/models \
  -H "Authorization: Bearer replace-this-with-a-long-random-secret"
```

For chat completions:

```bash
curl https://your-public-url.example/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer replace-this-with-a-long-random-secret" \
  -d '{
    "model": "SocratTeachLLM",
    "messages": [{"role": "user", "content": "What is photosynthesis?"}],
    "max_tokens": 200
  }'
```

## When this is a good fit

Use the tunnel approach if:

- you want the cheapest option
- your 3070 machine will stay powered on
- slower first-token latency is acceptable
- occasional home-network interruptions are okay

## When to use a cloud GPU instead

Use Runpod, Vast.ai, Modal, or another GPU host if:

- the service needs to stay up reliably
- you want better uptime than a home PC can provide
- you do not want to expose your home network
- you expect more than light personal/demo traffic

For this repo, a hosted GPU path mainly changes infrastructure, not app code:

- install Poetry + repo dependencies
- install the correct CUDA PyTorch wheel
- download `yuanpan/SocratTeachLLM`
- run `poetry run serve-teacher`
- place a reverse proxy in front if you need HTTPS/custom auth

## Important constraints

- An RTX 3070 is fine for personal use, demos, and light traffic, but not for
  many concurrent users.
- This server does not support streaming responses today.
- Keep the API key secret if you expose the endpoint publicly.
- Prefer a tunnel or proxy over opening a raw home router port.
