# Video Upscaling

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
