# Backend Kernel Libraries

## cuDNN

```bash
# reuse the cudnn package from torch installation
export CUDNN_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:$LD_LIBRARY_PATH"

cd cudnn-frontend
uv pip install -r ./requirements.txt
uv pip install pre-commit
pip install -e . --no-deps
```
