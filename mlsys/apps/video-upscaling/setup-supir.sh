#!/bin/bash

# get current directory
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get util functions
source "${CURR_DIR}/../../../scripts/common.sh"

splitLine
MY_VENV="supir"
CURR_VENV=$(getVenv)
if [ "${CURR_VENV}" != "${MY_VENV}" ]; then
    warnMsg "Please activate the ${MY_VENV} virtual environment first."
    exit 1
fi

splitLine
infoMsg "Installing dependencies ..."
pip3 install -r "${CURR_DIR}/SUPIR/requirements.txt"

splitLine
infoMsg "Downloading checkpoints ..."
pushd ckpts/

# SDXL CLIP Encoder-1
if ! [ -d "clip-vit-large-patch14" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/openai/clip-vit-large-patch14
fi
pushd clip-vit-large-patch14/
git lfs pull --include "model.safetensors"
git lfs pull --include "pytorch_model.bin"
popd

# SDXL CLIP Encoder-2
if ! [ -d "CLIP-ViT-bigG-14-laion2B-39B-b160k" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
fi
pushd CLIP-ViT-bigG-14-laion2B-39B-b160k/
git lfs pull --include "open_clip_pytorch_model.bin"
popd

# SDXL base 1.0_0.9vae
if ! [ -d "stable-diffusion-xl-base-1.0" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
fi
pushd stable-diffusion-xl-base-1.0/
git lfs pull --include "sd_xl_base_1.0_0.9vae.safetensors"
popd

# LLaVA CLIP
if ! [ -d "clip-vit-large-patch14-336" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/openai/clip-vit-large-patch14-336
fi
pushd clip-vit-large-patch14-336/
git lfs pull --include "pytorch_model.bin"
popd

# LLaVa v1.5 13B
if ! [ -d "llava-v1.5-13b" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/liuhaotian/llava-v1.5-13b
fi
pushd llava-v1.5-13b/
git lfs pull
popd

# Done
popd
