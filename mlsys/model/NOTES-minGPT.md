# minGPT

- [minGPT](#mingpt)
  - [Overview](#overview)
  - [Library Installation](#library-installation)

## Overview

GPT implementation in 300 lines of code in [mingpt/model.py](minGPT/mingpt/model.py).
- Feed a sequence of indices into a transformer.
- Output a probability distribution over the next index in the sequence.

Key files include:
- [mingpt/model.py](minGPT/mingpt/model.py): actual transformer model definition
- [mingpt/bpe.py](minGPT/mingpt/bpe.py): Byte Pair Encoder that translates b/w text & sequence of integers
- [mingpt/trainer.py](minGPT/mingpt/trainer.py): pytorch code that trains the model.

## Library Installation

```bash
pip3 install -e minGPT/
```
