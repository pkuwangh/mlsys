# SGLang

## Install SGLang

```bash
# use virtualenv and install system deps
source source_me_install_deps.sh

# install from source
cd sglang
pip install -e "python"

# quick check
cd ../
./check_cuda.py
```

## Quick Start

```bash
cd quickstart/

# online serving
python3 -m sglang.launch_server \
    --model-path ../models/Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --host 0.0.0.0 --port 8000 --log-level warning

curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a joke."}]
  }'

# offline inference
./s01-offline-batch-inference.py
```

## Diffusion Models

```bash
cd sglang
pip install -e "python[diffusion]"

# text to image
sglang generate --model-path=models/Qwen/Qwen-Image --prompt='A little kid playing basketball' --width=720 --height=720 --save-output
# text to video
sglang generate --model-path=models/Wan-AI/Wan2.1-T2V-14B-Diffusers --prompt "A little kid playing basketball" --720p --save-output
```
