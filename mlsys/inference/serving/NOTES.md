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

Triples are P10 / P50 / P90. All runs use synthetic K5/AL3.62 and one-hour measurements; speculative K is listed below.
`ull` (ultra low latency) denotes the `latency_specialized` decode backend.
Dynamo 1P2D uses one TP8/DCP8/EP8 prefill worker and two TP8/DCP8 decode workers without EP. Its disabled parsers differ from the aggregate runs.

| AIPerf metric | C1 | C2 | C2 | C4 | C8 | C16 | C16 | C16 | C32 | C32 | C32 | C32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Configuration | TP16 / DCP16 | TP16 / DCP16 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 / EP8 | TP8 / DCP8 | TP8 / DCP8 / EP8 | TP4 / DP2 / DCP4 / EP8 | 1P2D / TP8 / DCP8 |
| Kernel selection | `ull` | `ull` | `ull` | `ull` | `ull` | `ull` | `auto` | `auto` | `auto` | `auto` | `auto` | `auto` |
| GPUs | 16 | 16 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 24 |
| Speculative K | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| Tool/reasoning parsers | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Kimi | Disable |
| Max sequences | 8 | 8 | 8 | 8 | 8 | 16 | 16 | 16 | 32 | 32 | 16 per replica | P:32; D:16 each |
| ISL (tokens) | 71.4K / 341K / 556K | 24.7K / 135K / 473K | 24.7K / 136K / 423K | 24.3K / 119K / 521K | 45.3K / 107K / 354K | 37.1K / 95.5K / 224K | 36.5K / 92.9K / 205K | 36.5K / 92.4K / 200K | 37.6K / 94.5K / 229K | 38.7K / 96.7K / 236K | 30.5K / 72.9K / 148K | 43.6K / 127K / 335K |
| OSL (tokens) | 117 / 862 / 5.09K | 93 / 458 / 4.74K | 93 / 442 / 4.66K | 96 / 388 / 3.76K | 119 / 418 / 2.89K | 105 / 316 / 2.26K | 105 / 303 / 2.18K | 105 / 295 / 2.16K | 102 / 310 / 2.1K | 103 / 318 / 2.16K | 104 / 261 / 1.44K | 108 / 424 / 2.65K |
| Total throughput (tok/s/GPU) | 1.62K | 1.6K | 2.97K | 4.13K | 6.97K | 9.78K | 8.64K | 8.59K | 9.7K | 10.3K | 2.74K | 8.56K |
| Input token throughput (tok/s) | 25.7K | 25.4K | 23.6K | 32.8K | 55.3K | 77.7K | 68.6K | 68.2K | 77.1K | 82.2K | 21.8K | 204K |
| Output token throughput (tok/s) | 168 | 190 | 188 | 226 | 408 | 568 | 517 | 512 | 549 | 584 | 143 | 1.23K |
| Output token throughput (tok/s/GPU) | 10.5 | 11.9 | 23.5 | 28.3 | 51 | 71 | 64.6 | 64 | 68.6 | 73 | 17.9 | 51.3 |
| Output speed (tok/s/user) | **462** / 509 / 534 | **414** / 509 / 567 | **291** / 444 / 500 | **236** / 410 / 491 | **119** / 254 / 436 | **49.8** / 118 / 282 | **47.3** / 80 / 169 | **47** / 78.8 / 156 | **16.6** / 38.3 / 74.6 | **15.6** / 43.1 / 78.1 | **6.1** / 8.3 / 36.8 | **107** / 167 / 222 |
| Request latency (s) | 0.7 / 1.4 / 7 | 0.6 / 1.2 / 5.9 | 0.78 / 1.6 / 6.9 | 0.85 / 1.8 / 7.7 | 0.96 / 2.2 / 7.9 | 1.1 / 3 / 11.9 | 1.4 / 3.4 / 13.5 | 1.5 / 3.3 / 13.6 | 2.2 / 7.2 / 47.9 | 2 / 7.2 / 49 | 5.7 / 88.6 / 221 | 1.6 / 4.5 / 18.9 |
| Full decode time (s) | 0.21 / 1.8 / 10.4 | 0.17 / 0.98 / 9.6 | 0.21 / 1.2 / 11.5 | 0.22 / 1.3 / 11.1 | 0.41 / 2 / 12.6 | 0.56 / 3.1 / 20.7 | 0.91 / 4.1 / 27.2 | 0.98 / 4.1 / 29.2 | 1.6 / 9 / 77 | 1.6 / 8.9 / 80 | 4.7 / 23.2 / 191 | 0.6 / 2.7 / 16.4 |
| TTFT (s) | 0.44 / 0.78 / 1.2 | 0.41 / 0.75 / 1.4 | 0.5 / 0.93 / 2 | 0.5 / 1.1 / 2.1 | 0.5 / 1.1 / 2.1 | 0.54 / 1.2 / 2.5 | 0.56 / 1.2 / 2.2 | 0.57 / 1.2 / 2.4 | 0.75 / 1.8 / 12.3 | 0.67 / 1.6 / 10.6 | 1.9 / 66 / 161 | 0.61 / 1.2 / 3.2 |
| TPOT (ms) | 1.9 / 2 / 2.2 | 1.8 / 2 / 2.4 | 2 / 2.3 / 3.4 | 2 / 2.4 / 4.2 | 2.3 / 3.9 / 8.4 | 3.5 / 8.5 / 20.1 | 5.9 / 12.5 / 21.1 | 6.4 / 12.7 / 21.3 | 13.4 / 26.1 / 60.3 | 12.8 / 23.2 / 63.9 | 27.2 / 121 / 164 | 4.5 / 6 / 9.3 |

Output speed and TPOT use full-response metrics. Bold output speeds are P10 speed (approximately P90 interactivity).

| vLLM metric | C1 | C2 | C2 | C4 | C8 | C16 | C16 | C16 | C32 | C32 | C32 | C32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Configuration | TP16 / DCP16 | TP16 / DCP16 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 | TP8 / DCP8 / EP8 | TP8 / DCP8 | TP8 / DCP8 / EP8 | TP4 / DP2 / DCP4 / EP8 | 1P2D / TP8 / DCP8 |
| Kernel selection | `ull` | `ull` | `ull` | `ull` | `ull` | `ull` | `auto` | `auto` | `auto` | `auto` | `auto` | `auto` |
| Prefill excluding cached tokens | 766 / 2.21K / 10.1K | 383 / 2K / 10.2K | 380 / 2K / 10.1K | 344 / 1.73K / 8.4K | 374 / 1.76K / 7.66K | 341 / 1.52K / 8.73K | 340 / 1.5K / 8.91K | 334 / 1.47K / 8.83K | 368 / 1.68K / 15.6K | 377 / 1.69K / 15.4K | R0: 1.62K / 57.7K / 164K; R1: 926 / 52.6K / 170K | P: 394 / 1.72K / 6.97K |
| Prefix cache hit ratio | 98.1% | 97.4% | 97.2% | 97.5% | 97.3% | 95.8% | 95.8% | 95.7% | 89% | 89.4% | R0: 14.2%; R1: 18% | P: 92.2%; D0: 2.7%; D1: 4.3% |
| KV cache utilization (%) | 0 / 0 / 0.63 | 0 / 0 / 0.62 | 0 / 0 / 3 | 0 / 1.3 / 4.1 | 0 / 3.3 / 7.7 | 1.7 / 10.3 / 20.4 | 2.9 / 12.9 / 20.5 | 3.3 / 13.3 / 20.9 | 22.4 / 36.9 / 49.5 | 20.2 / 36 / 50.3 | R0: 54.6 / 81.4 / 94.7; R1: 42.6 / 83.3 / 96.4 | P: 0 / 1.9 / 6.2; D0: 2.3 / 6.8 / 12.8; D1: 2.1 / 6.5 / 12.8 |
| Avg Batched tokens | 14.4 | 17.2 | 18.2 | 20.2 | 33.1 | 106 | 135 | 146 | 1.14K | 1.08K | R0: 2.19K; R1: 2.07K | P: 5.25K; D0: 12.8; D1: 12.5 |
| Batched tokens | 2.5 / 3.7 / 5 | 1.8 / 3.8 / 5.9 | 1.5 / 3.9 / 6.8 | 1.6 / 4.2 / 7.4 | 2 / 5.8 / 13.9 | 3.1 / 12.1 / 34.5 | 4.3 / 17.7 / 44 | 5.5 / 20.8 / 46.1 | 37 / 71.9 / 121 | 35.4 / 67.4 / 120 | R0: 8.8 / 17.5 / 28.4; R1: 7.7 / 16.9 / 27.5 | P: 428 / 1.85K / 8.66K; D0: 3.7 / 11.2 / 25.4; D1: 3.7 / 11.2 / 24.7 |
| Avg Running requests | 0.36 | 0.45 | 0.56 | 0.8 | 2.1 | 6.2 | 7.4 | 7.5 | 21.7 | 21.2 | R0: 11.6; R1: 10.9 | P: 1; D0: 4; D1: 3.8 |
| Running requests | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 1 / 2 | 0 / 2 / 4 | 1 / 6 / 12 | 2 / 8 / 13 | 2 / 8 / 12 | 14 / 21 / 31 | 13 / 21 / 31 | R0: 8 / 12 / 14; R1: 7 / 11 / 14 | P: 0 / 1 / 2; D0: 1 / 4 / 7; D1: 1 / 3 / 7 |
| Avg waiting requests — capacity | 0.0006 | 0.0028 | 0.0033 | 0.0048 | 0.025 | 0.14 | 0.11 | 0.17 | 2.6 | 2.6 | R0: 7.2; R1: 7.7 | P: 0.25; D0: 0.0089; D1: 0.0064 |

Running requests come from `vllm:num_requests_running`: a sampled batch-size proxy including idle periods.
Capacity waiting is the mean of `vllm:num_requests_waiting_by_reason{reason="capacity"}`; deferred requests are excluded.
KV cache utilization is `vllm:kv_cache_usage_perc × 100`; evictable cached-prefix blocks are excluded.

