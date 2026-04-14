# Setup Commands — RTX 5090 Rig

## 1. Install system packages
```bash
sudo pacman -Syu --noconfirm
sudo pacman -S --noconfirm python-pip python-poetry
```

## 2. Install Python dependencies
```bash
cd ~/Documents/scu/CSEN-346/csen-346
pip install openai vllm transformers accelerate huggingface_hub rouge-score sacrebleu tqdm rich wandb
```

## 3. Download SocratTeachLLM (~19GB)
```bash
huggingface-cli download yuanpan/SocratTeachLLM --local-dir ~/hf_models/SocratTeachLLM
```

## 4. Configure environment
```bash
cp .env.example .env
# Edit .env — set your real CONSULTANT_API_KEY (OpenAI key for GPT-4o)
```

## 5. Start vLLM server (separate terminal / tmux pane)
```bash
./scripts/serve_socratteachllm.sh
```

## 6. Smoke test (3 dialogues)
```bash
python3 -m src.project.kele test --n 3 --output results/test
```

## 7. Full overnight evaluation (680 dialogues)
```bash
./scripts/run_eval.sh
```
