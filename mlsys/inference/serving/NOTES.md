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

Triples are P10 / P50 / P90. All runs use synthetic AL 3.62 and one-hour measurements; speculative K is listed below. GPU-normalized throughput counts all allocated GPUs, including prefill.
Decode graphs cover every request count from 1 to max sequences. Graph capture limits: C1–C8: 48 rows; C16: 96; C32 TP8: 192; C32 DP2: 96 per replica; 1P2D: P192 / D96.
Dynamo 1P2D uses one TP8/DCP8/EP8 prefill worker and two TP8/DCP8 decode workers without EP. Its disabled parsers differ from the aggregate runs.

| AIPerf metric | C1 | C2 | C2 | C4 | C8 | C16 | C16 | C16 | C32 | C32 | C32 | C32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Configuration | TP16 / DCP16 | TP16 / DCP16 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 / EP8 | TP8 / DCP8 | TP8 / DCP8 / EP8 | TP4 / DP2 / DCP4 / EP8 | 1P2D / TP8 / DCP8 |
| Kernel selection | `latency_specialized` | `latency_specialized` | `latency_specialized` | `latency_specialized` | `latency_specialized` | `latency_specialized` | `auto` | `auto` | `auto` | `auto` | `auto` | `auto` |
| GPUs | 16 | 16 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 24 |
| Speculative K | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| Tool/reasoning parsers | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Disabled |
| Max sequences | 8 | 8 | 8 | 8 | 8 | 16 | 16 | 16 | 32 | 32 | 16 per replica | P:32; D:16 each |
| ISL (tokens) | 71K / 341K / 556K | 25K / 135K / 473K | 25K / 136K / 423K | 24K / 119K / 521K | 45K / 107K / 354K | 37K / 96K / 224K | 36K / 93K / 205K | 36K / 92K / 200K | 38K / 94K / 229K | 39K / 97K / 236K | 30K / 73K / 148K | 44K / 127K / 335K |
| OSL (tokens) | 117 / 862 / 5.1K | 93 / 458 / 4.7K | 93 / 442 / 4.7K | 96 / 388 / 3.8K | 119 / 418 / 2.9K | 105 / 316 / 2.3K | 105 / 303 / 2.2K | 105 / 295 / 2.2K | 102 / 310 / 2.1K | 103 / 318 / 2.2K | 104 / 261 / 1.4K | 108 / 424 / 2.7K |
| Total throughput (tok/s/GPU) | 1.6K | 1.6K | 3K | 4.1K | 7K | 9.8K | 8.6K | 8.6K | 9.7K | 10K | 2.7K | 8.6K |
| Input token throughput (tok/s) | 26K | 25K | 24K | 33K | 55K | 78K | 69K | 68K | 77K | 82K | 22K | 200K |
| Output token throughput (tok/s) | 168.28 | 190.26 | 187.91 | 226.23 | 408.34 | 568.16 | 516.59 | 512.33 | 548.65 | 584.02 | 143.18 | 1,231.38 |
| Output token throughput (tok/s/GPU) | 10.52 | 11.89 | 23.49 | 28.28 | 51.04 | 71.02 | 64.57 | 64.04 | 68.58 | 73.00 | 17.90 | 51.31 |
| Output speed (tok/s/user) | **462.00** / 509.41 / 534.01 | **414.21** / 508.72 / 567.28 | **291.26** / 443.50 / 500.28 | **236.17** / 409.78 / 491.37 | **118.67** / 254.16 / 436.01 | **49.81** / 117.99 / 281.88 | **47.30** / 80.04 / 169.21 | **47.02** / 78.83 / 155.96 | **16.57** / 38.30 / 74.58 | **15.64** / 43.14 / 78.13 | **6.11** / 8.27 / 36.79 | **107.20** / 167.43 / 222.15 |
| Request latency (s) | 0.7 / 1.4 / 7 | 0.6 / 1.2 / 5.9 | 0.78 / 1.6 / 6.9 | 0.85 / 1.8 / 7.7 | 0.96 / 2.2 / 7.9 | 1.1 / 3 / 12 | 1.4 / 3.4 / 13 | 1.5 / 3.3 / 14 | 2.2 / 7.2 / 48 | 2 / 7.2 / 49 | 5.7 / 89 / 220 | 1.6 / 4.5 / 19 |
| Full decode time (s) | 0.21 / 1.8 / 10 | 0.17 / 0.98 / 9.6 | 0.21 / 1.2 / 12 | 0.22 / 1.3 / 11 | 0.41 / 2 / 13 | 0.56 / 3.1 / 21 | 0.91 / 4.1 / 27 | 0.98 / 4.1 / 29 | 1.6 / 9 / 77 | 1.6 / 8.9 / 80 | 4.7 / 23 / 190 | 0.6 / 2.7 / 16 |
| TTFT (s) | 0.44 / 0.78 / 1.2 | 0.41 / 0.75 / 1.4 | 0.5 / 0.93 / 2 | 0.5 / 1.1 / 2.1 | 0.5 / 1.1 / 2.1 | 0.54 / 1.2 / 2.5 | 0.56 / 1.2 / 2.2 | 0.57 / 1.2 / 2.4 | 0.75 / 1.8 / 12 | 0.67 / 1.6 / 11 | 1.9 / 66 / 160 | 0.61 / 1.2 / 3.2 |
| TPOT (ms) | 1.9 / 2 / 2.2 | 1.8 / 2 / 2.4 | 2 / 2.3 / 3.4 | 2 / 2.4 / 4.2 | 2.3 / 3.9 / 8.4 | 3.5 / 8.5 / 20 | 5.9 / 12 / 21 | 6.4 / 13 / 21 | 13 / 26 / 60 | 13 / 23 / 64 | 27 / 120 / 160 | 4.5 / 6 / 9.3 |

Output speed and TPOT use full-response metrics. Bold output speeds are P10 speed (approximately P90 interactivity).

| vLLM metric | C1 | C2 | C2 | C4 | C8 | C16 | C16 | C16 | C32 | C32 | C32 | C32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Configuration | TP16 / DCP16 | TP16 / DCP16 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 / EP8 | TP8 / DCP8 | TP8 / DCP8 / EP8 | TP4 / DP2 / DCP4 / EP8 | 1P2D / TP8 / DCP8 |
| Kernel selection | `latency_specialized` | `latency_specialized` | `latency_specialized` | `latency_specialized` | `latency_specialized` | `latency_specialized` | `auto` | `auto` | `auto` | `auto` | `auto` | `auto` |
| Prefill excluding cached tokens | 770 / 2.2K / 10K | 380 / 2K / 10K | 380 / 2K / 10K | 340 / 1.7K / 8.4K | 370 / 1.8K / 7.7K | 340 / 1.5K / 8.7K | 340 / 1.5K / 8.9K | 330 / 1.5K / 8.8K | 370 / 1.7K / 16K | 380 / 1.7K / 15K | R0: 1.6K / 58K / 160K; R1: 930 / 53K / 170K | P: 390 / 1.7K / 7K |
| Prefix cache hit ratio | 98.1% | 97.4% | 97.2% | 97.5% | 97.3% | 95.8% | 95.8% | 95.7% | 89.0% | 89.4% | R0: 14.2%; R1: 18.0% | P: 92.2%; D0: 2.7%; D1: 4.3% |
| KV cache utilization (%) | 0 / 0 / 0.63 | 0 / 0 / 0.62 | 0 / 0 / 3 | 0 / 1.3 / 4.1 | 0 / 3.3 / 7.7 | 1.7 / 10 / 20 | 2.9 / 13 / 20 | 3.3 / 13 / 21 | 22 / 37 / 50 | 20 / 36 / 50 | R0: 55 / 81 / 95; R1: 43 / 83 / 96 | P: 0 / 1.9 / 6.2; D0: 2.3 / 6.8 / 13; D1: 2.1 / 6.5 / 13 |
| Avg Batched tokens | 14 | 17 | 18 | 20 | 33 | 110 | 130 | 150 | 1.1K | 1.1K | R0: 2.2K; R1: 2.1K | P: 5.2K; D0: 13; D1: 13 |
| Batched tokens | 2.5 / 3.7 / 5 | 1.8 / 3.8 / 5.9 | 1.5 / 3.9 / 6.8 | 1.6 / 4.2 / 7.4 | 2 / 5.8 / 14 | 3.1 / 12 / 35 | 4.3 / 18 / 44 | 5.5 / 21 / 46 | 37 / 72 / 120 | 35 / 67 / 120 | R0: 8.8 / 18 / 28; R1: 7.7 / 17 / 27 | P: 430 / 1.9K / 8.7K; D0: 3.7 / 11 / 25; D1: 3.7 / 11 / 25 |
| Avg Running requests | 0.36 | 0.45 | 0.56 | 0.80 | 2.14 | 6.23 | 7.44 | 7.55 | 21.67 | 21.24 | R0: 11.56; R1: 10.88 | P: 1.00; D0: 3.95; D1: 3.84 |
| Running requests | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 1 / 2 | 0 / 2 / 4 | 1 / 6 / 12 | 2 / 8 / 13 | 2 / 8 / 12 | 14 / 21 / 31 | 13 / 21 / 31 | R0: 8 / 12 / 14; R1: 7 / 11 / 14 | P: 0 / 1 / 2; D0: 1 / 4 / 7; D1: 1 / 3 / 7 |
| Avg waiting requests — capacity | 0.0006 | 0.0028 | 0.0033 | 0.0048 | 0.025 | 0.14 | 0.11 | 0.17 | 2.6 | 2.6 | R0: 7.2; R1: 7.7 | P: 0.25; D0: 0.0089; D1: 0.0064 |

Running requests come from `vllm:num_requests_running`: a sampled batch-size proxy including idle periods.
Capacity waiting is the mean of `vllm:num_requests_waiting_by_reason{reason="capacity"}`; deferred requests are excluded.
KV cache utilization is `vllm:kv_cache_usage_perc × 100`; evictable cached-prefix blocks are excluded.

