# Inference Tutorial

## Setup environment

```bash
# install packages in a virtualenv
micromamba create -n mlsys_inference_tutorial -c conda-forge python=3.10 pip=23.2 -y
micromamba activate mlsys_inference_tutorial
pip3 install -r requirements.txt
pip3 install ipykernel

# download model
cd models/
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/openai-community/gpt2
cd gpt2/
git lfs pull --include model.safetensors
cd ../../
```
