# Production Inference Serving

## Workload Behavior

- **agentic trace**
  - ordered turns with growing session context
  - real dependency plus recorded inter-turn delays
  - concurrency describes live sessions instead of in-flight requests
- **subagents**:
  - after parent request finishes, one or more subagent branches can run concurrently
  - there may be partial cached prefix, depending on how agent construct the child's input
- **This C1 run's stats** (P10/P50/P90)
  - ISL: 71K / 341K / 554K
  - OSL: 117 / 862 / 5.1K

## vLLM engine configs and stats

Triples are P10 / P50 / P90.

| AIPerf metric | C1 | C2 | C2 |
| --- | ---: | ---: | ---: |
| Configuration | TP16 / DCP16 | TP16 / DCP16 | TP8 / DCP8 |
| Total throughput (tok/s/GPU) | 1.6K | 1.6K | 2.9K |
| Input token throughput (tok/s) | 26K | 25K | 23K |
| Output token throughput (tok/s) | 170 | 190 | 180 |
| Output speed (tok/s/user) | **460** / 510 / 540 | **410** / 510 / 560 | **320** / 440 / 500 |
| Request latency (s) | 0.69 / 1.5 / 6.9 | 0.63 / 1.2 / 5.7 | 0.72 / 1.5 / 6.7 |
| Full decode time (s) | 0.21 / 1.8 / 10 | 0.18 / 1.1 / 10 | 0.20 / 1.1 / 11 |
| TTFT (s) | 0.44 / 0.76 / 1.3 | 0.43 / 0.74 / 1.4 | 0.46 / 0.99 / 1.9 |
| TPOT (ms) | 1.8 / 2.0 / 2.2 | 1.8 / 2.0 / 2.4 | 2.0 / 2.3 / 3.1 |

Bold output speeds are P10 speed, corresponding to P90 interactivity.

| vLLM metric | C1 | C2 | C2 |
| --- | ---: | ---: | ---: |
| Configuration | TP16 / DCP16 | TP16 / DCP16 | TP8 / DCP8 |
| Prefill excluding cached tokens | 760 / 2.2K / 10K | 380 / 2.0K / 10K | 370 / 2.0K / 11K |
| Prefix cache hit ratio | 98% | 97% | 97% |
| Avg Batched tokens | 14 | 17 | 18 |
| Batched tokens | 2.5 / 3.8 / 5.0 | 1.8 / 3.9 / 5.9 | 1.7 / 3.8 / 6.0 |
| Running requests | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 |

Running requests come from `vllm:num_requests_running`: a sampled batch-size proxy including idle periods.

