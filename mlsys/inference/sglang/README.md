# SGLang

- [SGLang](#sglang)
  - [Install SGLang and SGLang-Omni](#install-sglang-and-sglang-omni)
  - [SGLang Instructions](#sglang-instructions)
    - [SGLang Quick Start](#sglang-quick-start)
    - [SGLang-Omni Quick Start](#sglang-omni-quick-start)
    - [Diffusion Models](#diffusion-models)

## Install SGLang and SGLang-Omni

```bash
# use virtualenv and install system deps
source source_me_install_deps.sh

# install sglang
cd sglang
uv pip freeze | grep -E '^(torch|torchvision|torchaudio)==' | tee /tmp/sglang-torch-overrides.txt
uv pip install -e ./python --override /tmp/sglang-torch-overrides.txt


# install sglang-omni
cd sglang-omni
uv pip install --prerelease=allow -e . --override /tmp/sglang-torch-overrides.txt
```

## SGLang Instructions

### SGLang Quick Start

```bash
cd quickstart/

# online serving
python3 -m sglang.launch_server \
    --model-path ../models/Qwen/Qwen3.8-27B \
    --host 0.0.0.0 --port 8000 --log-level warning

curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a joke."}]
  }'

# offline inference
./s01-offline-batch-inference.py
```

### SGLang-Omni Quick Start

```bash
uv pip install --no-deps "qwen-tts==0.1.1"
```

### Diffusion Models

```bash
cd sglang
uv pip install -e "python[diffusion]" --override /tmp/sglang-torch-overrides.txt

# text to image
sglang generate --model-path=models/Qwen/Qwen-Image --prompt='A little kid playing basketball' --width=720 --height=720 --save-output
# text to video
sglang generate --model-path=models/Wan-AI/Wan2.1-T2V-14B-Diffusers --prompt "A little kid playing basketball" --720p --save-output
```
