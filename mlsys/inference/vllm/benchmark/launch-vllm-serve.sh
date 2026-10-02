#!/usr/bin/env bash

QWEN_VIDEO_TEMPORAL_MODE=framewise \
vllm serve ./models/video_captioner/qwen35-27b-exp3-gpt55-seed21-general500k-yatian500k-fps2-framewise-step16193-20260812074816-vllm-v1 \
    --host 127.0.0.1 \
    --port 8089 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --api-server-count 8 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 65536 \
    --max-num-batched-tokens 131072 \
    --max-num-seqs 32 \
    --no-enable-prefix-caching \
    --reasoning-parser qwen3 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-gb 0 \
    --limit-mm-per-prompt.video 1 \
    --limit-mm-per-prompt.image 0 \
    --media-io-kwargs '{"video":{"num_frames":-1,"fps":-1}}' \
    --mm-processor-kwargs '{"fps":2,"min_frames":4,"max_frames":256,"min_pixels":4096,"max_pixels":49152000}' \
    --allowed-local-media-path "$(realpath ./data-samples)"
