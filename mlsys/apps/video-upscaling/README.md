# Video Upscaling

## SUPIR

Installation & download checkpoints

```bash
./setup-supir.sh
```

Two more checkpoints need to be downloaded from [google drive](https://github.com/Fanghua-Yu/SUPIR?tab=readme-ov-file#models-we-provided).

Usage for images

```bash
# Note this cuda device index should align with pytorch's device order, which is fastest first
# It can be different than nvidia-smi's order, which is based on PCIe ID.
export CUDA_VISIBLE_DEVICES=0
python3 test.py --img_dir input/ --save_dir . --upscale 2 --SUPIR_sign Q
python3 test.py --img_dir input/ --save_dir . --upscale 2 --SUPIR_sign F --s_cfg 4.0 --linear_CFG
```

## RealBasicVSR

Installation

```bash
./setup-real-basic-vsr.sh
```

Download [checkpoint](https://github.com/ckkelvinchan/RealBasicVSR?tab=readme-ov-file#inference)

```bash
pushd ckpts/
wget https://www.dropbox.com/s/eufigxmmkv5woop/RealBasicVSR.pth
popd
```

Usage

```bash
pushd RealBasicVSR/
python3 inference_realbasicvsr.py configs/realbasicvsr_x4.py ../ckpts/RealBasicVSR.pth ./data/demo_001.mp4 output.mp4 --max_seq_len 4 --fps 12.5
popd
```
