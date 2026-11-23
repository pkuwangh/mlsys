# Production Inference Serving

## Workload Behavior

- **agentic trace**
  - ordered turns with growing session context
  - real dependency plus recorded inter-turn delays
  - concurrency describes live sessions instead of in-flight requests
- **subagents**:
  - after parent request finishes, one or more subagent branches can run concurrently
  - there may be partial cached prefix, depending on how agent construct the child's input

## vLLM engine configs and stats

Percentile pairs are P50 / P90. All runs use one-hour measurements. Speculative decoding uses synthetic K5/AL3.62 unless listed as disabled.
`ull` (ultra low latency) denotes the `latency_specialized` decode backend.

| AIPerf metric | C1 | C2 | C2 | C4 | C8 | C16 | C16 | C16 | C32 | C32 | C32 | C32 | C48 | C48 | C48 | C48 | C64 | C64 | C64 | C80 | C80 | C80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Configuration | TP16 / DCP16 | TP16 / DCP16 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 / EP8 | TP8 / DCP8 | TP8 / DCP8 / EP8 | TP4 / DP2 / DCP4 / EP8 | 1P2D / TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 / CPU offload | TP8 / DCP8 | 1P2D / TP8 / DCP8 | 1P2D / TP8 / DCP8 | 1P2D / TP8 / DCP8 | 1P2D / TP8 / DCP8 | 1P2D / TP8 / DCP8 | 1P2D / TP8 / DCP8 | 1P2D / TP8 / DCP8 |
| Host KV cache per node (GiB RAM / SSD) | — | — | — | — | — | — | — | — | — | — | — | 640 / 0 | — | 320 / 0 | — | 640 / 0 | 640 / 0 | 672 / 0 | 640 / 640 | 640 / 0 | 672 / 0 | 640 / 640 |
| Kernel selection | `ull` | `ull` | `ull` | `ull` | `ull` | `ull` | `auto` | `auto` | `auto` | `auto` | `auto` | P: `auto`; D: `ull` | `auto` | `auto` | `auto` | P: `auto`; D: `auto` | P: `auto`; D: `auto` | P: `auto`; D: `auto` | P: `auto`; D: `auto` | P: `auto`; D: `auto` | P: `auto`; D: `auto` | P: `auto`; D: `auto` |
| GPUs | 16 | 16 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 24 | 8 | 8 | 8 | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| Speculative K | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | Disabled | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| Tool/reasoning parsers | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Disable | Kimi | Kimi | Kimi | Disable | Disable | Disable | Disable | Disable | Disable | Disable |
| Max sequences | 8 | 8 | 8 | 8 | 8 | 16 | 16 | 16 | 32 | 32 | 16 per replica | P:32; D:10 each | 48 | 48 | 48 | P:32; D:128 each | P:32; D:128 each | P:32; D:128 each | P:32; D:128 each | P:32; D:128 each | P:32; D:128 each | P:32; D:128 each |
| ISL (tokens) | 341K / 556K | 135K / 473K | 136K / 423K | 119K / 521K | 107K / 354K | 95.5K / 224K | 92.9K / 205K | 92.4K / 200K | 94.5K / 229K | 96.7K / 236K | 72.9K / 148K | 131K / 359K | 76.5K / 195K | 89.6K / 258K | 84.7K / 251K | 96.4K / 330K | 83.9K / 280K | 83.5K / 287K | 82.4K / 274K | 72.5K / 266K | 70.5K / 264K | 71.5K / 259K |
| OSL (tokens) | 862 / 5.09K | 458 / 4.74K | 442 / 4.66K | 388 / 3.76K | 418 / 2.89K | 316 / 2.26K | 303 / 2.18K | 295 / 2.16K | 310 / 2.1K | 318 / 2.16K | 261 / 1.44K | 427 / 2.66K | 296 / 1.5K | 358 / 2.13K | 308 / 1.76K | 425 / 2.48K | 459 / 2.31K | 448 / 2.33K | 459 / 2.35K | 442 / 2.19K | 460 / 2.14K | 447 / 2.27K |
| Total throughput (tok/s/GPU) | 1.62K | 1.6K | 2.97K | 4.13K | 6.97K | 9.78K | 8.64K | 8.59K | 9.7K | 10.3K | 2.74K | 9.91K | 4.79K | 14.7K | 9.93K | 11.5K | 9.49K | 11.8K | 8.21K | 8.14K | 10.6K | 6.27K |
| Input token throughput (tok/s) | 25.7K | 25.4K | 23.6K | 32.8K | 55.3K | 77.7K | 68.6K | 68.2K | 77.1K | 82.2K | 21.8K | 236K | 38.1K | 117K | 78.9K | 274K | 226K | 280K | 195K | 194K | 252K | 149K |
| Output token throughput (tok/s) | 168 | 190 | 188 | 226 | 408 | 568 | 517 | 512 | 549 | 584 | 143 | 1.37K | 243 | 787 | 508 | 1.83K | 1.78K | 2.13K | 1.57K | 1.64K | 2.17K | 1.32K |
| Output token throughput (tok/s/GPU) | 10.5 | 11.9 | 23.5 | 28.3 | 51 | 71 | 64.6 | 64 | 68.6 | 73 | 17.9 | 57 | 30.4 | 98.4 | 63.5 | 76.3 | 74 | 88.7 | 65.3 | 68.3 | 90.3 | 55.1 |
| Output speed (tok/s/user) | 509 / 534 | 509 / 567 | 444 / 500 | 410 / 491 | 254 / 436 | 118 / 282 | 80 / 169 | 78.8 / 156 | 38.3 / 74.6 | 43.1 / 78.1 | 8.3 / 36.8 | 265 / 385 | 7.9 / 57.5 | 27.1 / 50.1 | 16.8 / 29 | 128 / 175 | 108 / 174 | 98.9 / 150 | 118 / 181 | 119 / 169 | 90.2 / 148 | 135 / 186 |
| Request latency (s) | 1.4 / 7 | 1.2 / 5.9 | 1.6 / 6.9 | 1.8 / 7.7 | 2.2 / 7.9 | 3 / 11.9 | 3.4 / 13.5 | 3.3 / 13.6 | 7.2 / 47.9 | 7.2 / 49 | 88.6 / 221 | 2.7 / 11.3 | 50.4 / 204 | 9.4 / 48 | 21.5 / 118 | 4.8 / 25.9 | 12.6 / 92.2 | 7.9 / 58.9 | 29.8 / 74.8 | 50.9 / 99.2 | 25.3 / 77.8 | 65.6 / 120 |
| Full decode time (s) | 1.8 / 10.4 | 0.98 / 9.6 | 1.2 / 11.5 | 1.3 / 11.1 | 2 / 12.6 | 3.1 / 20.7 | 4.1 / 27.2 | 4.1 / 29.2 | 9 / 77 | 8.9 / 80 | 23.2 / 191 | 1.6 / 9.5 | 27.7 / 226 | 13.6 / 88.5 | 19.8 / 139 | 3.2 / 18.5 | 4 / 19.4 | 4.3 / 21.3 | 3.8 / 18.3 | 3.9 / 18.4 | 4.8 / 21.2 | 3.4 / 16.2 |
| TTFT (s) | 0.78 / 1.2 | 0.75 / 1.4 | 0.93 / 2 | 1.1 / 2.1 | 1.1 / 2.1 | 1.2 / 2.5 | 1.2 / 2.2 | 1.2 / 2.4 | 1.8 / 12.3 | 1.6 / 10.6 | 66 / 161 | 0.74 / 2 | 17.6 / 94.3 | 1.9 / 5.1 | 1.6 / 7.6 | 0.81 / 4.6 | 1.5 / 84.1 | 1 / 41.4 | 22.4 / 64.2 | 44.2 / 91 | 10.2 / 64.5 | 59 / 109 |
| TPOT (ms) | 2 / 2.2 | 2 / 2.4 | 2.3 / 3.4 | 2.4 / 4.2 | 3.9 / 8.4 | 8.5 / 20.1 | 12.5 / 21.1 | 12.7 / 21.3 | 26.1 / 60.3 | 23.2 / 63.9 | 121 / 164 | 3.8 / 5.2 | 126 / 215 | 36.9 / 61.8 | 59.6 / 105 | 7.8 / 9.6 | 9.2 / 11.8 | 10.1 / 11.8 | 8.5 / 10.9 | 8.4 / 12.4 | 11.1 / 13.5 | 7.4 / 10.3 |

Output speed and TPOT use full-response metrics.

| vLLM metric | C1 | C2 | C2 | C4 | C8 | C16 | C16 | C16 | C32 | C32 | C32 | C32 | C48 | C48 | C48 | C48 | C64 | C64 | C64 | C80 | C80 | C80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Configuration | TP16 / DCP16 | TP16 / DCP16 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 / EP8 | TP8 / DCP8 | TP8 / DCP8 / EP8 | TP4 / DP2 / DCP4 / EP8 | 1P2D / TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 / CPU offload | TP8 / DCP8 | 1P2D / TP8 / DCP8 | 1P2D / TP8 / DCP8 | 1P2D / TP8 / DCP8 | 1P2D / TP8 / DCP8 | 1P2D / TP8 / DCP8 | 1P2D / TP8 / DCP8 | 1P2D / TP8 / DCP8 |
| Host KV cache per node (GiB RAM / SSD) | — | — | — | — | — | — | — | — | — | — | — | 640 / 0 | — | 320 / 0 | — | 640 / 0 | 640 / 0 | 672 / 0 | 640 / 640 | 640 / 0 | 672 / 0 | 640 / 640 |
| Kernel selection | `ull` | `ull` | `ull` | `ull` | `ull` | `ull` | `auto` | `auto` | `auto` | `auto` | `auto` | P: `auto`; D: `ull` | `auto` | `auto` | `auto` | P: `auto`; D: `auto` | P: `auto`; D: `auto` | P: `auto`; D: `auto` | P: `auto`; D: `auto` | P: `auto`; D: `auto` | P: `auto`; D: `auto` | P: `auto`; D: `auto` |
| Speculative K | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | Disabled | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| Prefill excluding cached tokens | 2.21K / 10.1K | 2K / 10.2K | 2K / 10.1K | 1.73K / 8.4K | 1.76K / 7.66K | 1.52K / 8.73K | 1.5K / 8.91K | 1.47K / 8.83K | 1.68K / 15.6K | 1.69K / 15.4K | R0: 57.7K / 164K; R1: 52.6K / 170K | P: 1.74K / 6.83K | 2.6K / 123K | 2.12K / 17K | 1.61K / 9.54K | P: 1.8K / 7.95K | P: 2K / 9.62K | P: 1.88K / 8.53K | P: 2.13K / 17K | P: 2.04K / 14.5K | P: 2.17K / 10K | P: 2.31K / 26.9K |
| GPU cache hit ratio | 98.1% | 97.4% | 97.2% | 97.5% | 97.3% | 95.8% | 95.8% | 95.7% | 89.2% | 89.4% | R0: 14.5%; R1: 18.1% | P: 89.1% | 58.1% | 67.7% | 91.3% | P: 87.6% | P: 68.7% | P: 75.3% | P: 71.2% | P: 44.8% | P: 50.7% | P: 37.4% |
| CPU + SSD cache hit ratio | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | P: 8.3% | 0% | 26% | 0% | P: 8.6% | P: 22.7% | P: 18.9% | P: 16.3% | P: 42.4% | P: 40.2% | P: 44.7% |
| KV cache utilization (%) | 0 / 0.63 | 0 / 0.62 | 0 / 3 | 1.3 / 4.1 | 3.3 / 7.7 | 10.3 / 20.4 | 12.9 / 20.5 | 13.3 / 20.9 | 36.9 / 49.5 | 36 / 50.3 | R0: 81.4 / 94.7; R1: 83.3 / 96.4 | P: 1.9 / 7.5; D0: 3.9 / 8.2; D1: 3.9 / 8.6 | 68.1 / 75.9 | 61.8 / 72.6 | 29.2 / 32 | P: 1.8 / 7.3; D0: 8.8 / 14.5; D1: 9.1 / 16 | P: 4.5 / 21.9; D0: 8.9 / 19.2; D1: 7.8 / 20.3 | P: 3.4 / 18.6; D0: 14 / 23.4; D1: 13 / 20.9 | P: 5.6 / 20.7; D0: 6.2 / 14.6; D1: 6.1 / 17.1 | P: 10.5 / 26.1; D0: 5.9 / 20.7; D1: 6 / 18.8 | P: 12.4 / 29.3; D0: 9.4 / 26.7; D1: 9.4 / 26.5 | P: 10.1 / 24.3; D0: 4.5 / 11.4; D1: 4.5 / 11.5 |
| KV cache capacity (GiB/GPU) | 111 | 111 | 47.6 | 47.6 | 47.6 | 47.3 | 48.6 | 48.1 | 47.6 | 47.4 | R0: 22.4; R1: 22.4 | P: 32.2; D0: 54.1; D1: 54.1 | 47.1 | 47.1 | 51 | P: 31.9; D0: 49.8; D1: 49.8 | P: 31.9; D0: 49.8; D1: 49.8 | P: 31.9; D0: 49.9; D1: 49.9 | P: 31.9; D0: 49.9; D1: 49.9 | P: 31.9; D0: 49.9; D1: 49.9 | P: 31.9; D0: 49.9; D1: 49.9 | P: 31.9; D0: 49.9; D1: 49.9 |
| Token capacity (tokens/worker) | 88.3M | 88.3M | 19M | 19M | 19M | 18.9M | 19.4M | 19.2M | 19M | 18.9M | R0: 4.46M; R1: 4.46M | P: 12.8M; D0: 21.6M; D1: 21.6M | 18.8M | 18.8M | 28.5M | P: 14M; D0: 21.9M; D1: 21.9M | P: 14M; D0: 21.9M; D1: 21.9M | P: 14M; D0: 21.9M; D1: 21.9M | P: 14M; D0: 21.9M; D1: 21.9M | P: 14M; D0: 21.9M; D1: 21.9M | P: 14M; D0: 21.9M; D1: 21.9M | P: 14M; D0: 21.9M; D1: 21.9M |
| Avg Batched tokens | 14.4 | 17.2 | 18.2 | 20.2 | 33.1 | 106 | 135 | 146 | 1.14K | 1.08K | R0: 2.19K; R1: 2.07K | P: 5.14K; D0: 8.9; D1: 9 | 6.74K | 1.16K | 519 | P: 6.96K; D0: 23.1; D1: 22.7 | P: 17.1K; D0: 24.4; D1: 24.4 | P: 10.4K; D0: 32.7; D1: 32.3 | P: 33.8K; D0: 19.4; D1: 19.7 | P: 30.2K; D0: 21.4; D1: 21.3 | P: 21K; D0: 31.3; D1: 30.6 | P: 45.8K; D0: 16.5; D1: 15.7 |
| Batched tokens | 3.7 / 5 | 3.8 / 5.9 | 3.9 / 6.8 | 4.2 / 7.4 | 5.8 / 13.9 | 12.1 / 34.5 | 17.7 / 44 | 20.8 / 46.1 | 71.9 / 121 | 67.4 / 120 | R0: 17.5 / 28.4; R1: 16.9 / 27.5 | P: 1.82K / 7.77K; D0: 7.1 / 16; D1: 7.1 / 16.5 | 128 / 662 | 128 / 582 | 40.2 / 50.6 | P: 1.97K / 10.7K; D0: 21.9 / 41.9; D1: 21.5 / 40.4 | P: 2.53K / 54K; D0: 17.2 / 55.9; D1: 17.3 / 56 | P: 2.36K / 15K; D0: 33 / 61.5; D1: 31.6 / 61.8 | P: 5.04K / 142K; D0: 14.2 / 45.5; D1: 14.6 / 45.6 | P: 4.51K / 129K; D0: 14.7 / 52.2; D1: 14.8 / 50.4 | P: 3.83K / 86.7K; D0: 21.9 / 76.7; D1: 20.4 / 76.4 | P: 8.58K / 178K; D0: 12.6 / 32; D1: 11.4 / 31.3 |
| Avg Running requests | 0.36 | 0.45 | 0.56 | 0.8 | 2.1 | 6.2 | 7.4 | 7.5 | 21.7 | 21.2 | R0: 11.6; R1: 10.9 | P: 0.63; D0: 2.6; D1: 2.7 | 37.5 | 35.3 | 41.2 | P: 0.91; D0: 7.1; D1: 6.9 | P: 1.8; D0: 7.9; D1: 7.7 | P: 1.5; D0: 10.1; D1: 10.2 | P: 2.5; D0: 6.1; D1: 6.2 | P: 2.3; D0: 6.8; D1: 7 | P: 2.5; D0: 10.9; D1: 10.8 | P: 2.4; D0: 4.8; D1: 4.6 |
| Running requests | 0 / 1 | 0 / 1 | 0 / 1 | 1 / 2 | 2 / 4 | 6 / 12 | 8 / 13 | 8 / 12 | 21 / 31 | 21 / 31 | R0: 12 / 14; R1: 11 / 14 | P: 0 / 1; D0: 2 / 5; D1: 2 / 5 | 42 / 47 | 36 / 44 | 43 / 48 | P: 1 / 2; D0: 7 / 11; D1: 7 / 11 | P: 1 / 4; D0: 7 / 17; D1: 7 / 17 | P: 1 / 3; D0: 11 / 18; D1: 11 / 18 | P: 2 / 6; D0: 5 / 14; D1: 5 / 14 | P: 1 / 6; D0: 5 / 18; D1: 5 / 18 | P: 1 / 7; D0: 8 / 24; D1: 8 / 24 | P: 2 / 5; D0: 4 / 10; D1: 4 / 10 |
| Avg waiting requests — capacity | 0.0006 | 0.0028 | 0.0033 | 0.0048 | 0.025 | 0.14 | 0.11 | 0.17 | 2.6 | 2.6 | R0: 7.2; R1: 7.7 | P: 0.15; D0: 0.0045; D1: 0.0057 | 13.8 | 0.73 | 1 | P: 1.7; D0: 0.02; D1: 0.015 | P: 9.5; D0: 0.02; D1: 0.023 | P: 5.8; D0: 0.025; D1: 0.028 | P: 14.5; D0: 0.018; D1: 0.014 | P: 14; D0: 0.02; D1: 0.021 | P: 14.1; D0: 0.03; D1: 0.033 | P: 14.3; D0: 0.0098; D1: 0.011 |

Running requests come from `vllm:num_requests_running`: a sampled batch-size proxy including idle periods.
Capacity waiting is the mean of `vllm:num_requests_waiting_by_reason{reason="capacity"}`; deferred requests are excluded.
KV cache utilization is `vllm:kv_cache_usage_perc × 100`; evictable cached-prefix blocks are excluded.

Native CPU offload provides approximately 320 GiB RAM/node (80 GiB/rank): `kv-offloading-size=640` is the TP8-wide budget, with a 640 GiB logical mapping on each node.
