# nanoGPT

- [nanoGPT](#nanogpt)
  - [Quickstart with a tiny model](#quickstart-with-a-tiny-model)
  - [Reproduce GPT-2](#reproduce-gpt-2)

## Quickstart with a tiny model

Install deps

```bash
pip3 install -r requirements-nanoGPT.txt
```

Download data

```bash
# in the data dir, this creates a
# - train.bin: first 90% of data
# - val.bin: last 10% of data
python3 ./data/shakespeare_char/prepare.py
```

Train a baby GPT; this takes about 1.8GB GPU memory and 0.9GB system memory.

```bash
python3 train.py config/train_shakespeare_char.py
```

Generate samples

```bash
python3 sample.py --out_dir=out-shakespeare-char
```

## Reproduce GPT-2

First tokenize the dataset,
- load huggingface dataset, to ~/.cache/huggingface/datasets/openwebtext/
- split dataset into train/eval sets
- tokenize the dataset

```bash
python3 data/openwebtext/prepare.py
```

Train GPT2
- For GPT-2 (124M), consider a 8x A100 40GB node
- with DDP, it runs ~4 days and go down to loss of ~2.85.

```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

Init from GPT-2 checkpoints

```bash
# eval_gpt.py has init_from='gpt2'
python3 train.py config/eval_gpt2.py
```

Finetuning - continue training from a checkpoint

```bash
# also init_from='gpt2'
python3 train.py config/finetune_shakespeare.py
```

Sample/Inference

```bash
python3 sample.py --start="When you look in the mirror" \
    --num_samples=2 --max_new_tokens=100 \
    --init_from=gpt2
python3 sample.py --start="When you look in the mirror" \
    --num_samples=2 --max_new_tokens=100 \
    --out_dir=out-shakespeare
```