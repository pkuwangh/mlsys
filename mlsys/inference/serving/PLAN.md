# Kimi K3 / K3.1 Serving Study

Models: `moonshotai/Kimi-K3` and `moonshotai/Kimi-K3.1-Pro`.
Hardware: external GB200 / GB300 cluster. Start on GB300 to match published results.
Keep README empty until we have results.

## Published Targets

| Model | Benchmark | Published result |
| --- | --- | --- |
| K3 | [AgentX, September 8][k3-agentx] | 16 GB300, concurrency 48: 11.8K total tokens/s/GPU, P90 interactivity 62.7 tokens/s/user |
| K3 | [Fused decode, October 16][k3-fused] | Over 840 tokens/s at batch 1 on 16 GB300, DSpark with synthetic acceptance length 6; up to 1.8x AgentX interactivity |
| K3.1 Pro | [AgentX, November 4][k31] | GB300 frontier: 375.9 tokens/s/user at the low-latency end; 37.8K total tokens/s/GPU at about 54 tokens/s/user at the throughput end |

The K3.1 AgentX curve uses 4 draft tokens and synthetic acceptance length 3.36.
The two endpoints are different configurations, not simultaneous performance.
Total throughput includes cached input, uncached input, and output tokens.

## What to Study

Both models have 93 layers: 69 KDA and 24 MLA, with 896 routed experts and 16 selected per token.
K3.1 adds a top-2048 DSA indexer to the MLA layers.

| Change | Measure |
| --- | --- |
| K3 dense MLA vs. K3.1 sparse MLA | Decode time and memory traffic as context grows |
| TP vs. decode context parallelism (DCP) | KV capacity, decode latency, supported concurrency |
| Default vs. fused CuTe DSL decode | Layer time and e2e interactivity |
| DSpark and prefix caching | Acceptance length, TTFT, throughput under multi-turn traffic |

## Execution

1. Extract the exact image digest, checkpoint/draft revisions, launch commands, and concurrency points behind each target.
   Start with the [K3][recipe-k3] and [K3.1][recipe-k31] recipes; K3.1's basic GB300 setup is TP8 across two four-GPU trays.
2. Reproduce K3.1 AgentX first, then K3. Use the published TP/DCP/EP, KV dtype/offload, batch limits,
   graph settings, routing, and speculation settings. Do not reuse DeepSeek's configuration.
3. Run the pinned [AgentX harness][agentx] against the OpenAI-compatible frontend:
   model/tokenizer set to the selected checkpoint, full `semianalysis_cc_traces_weka_062126` corpus,
   seed 42, and the target's session-concurrency sweep.
4. Match the target's warmup and measurement duration; repeat selected points three times.
   For synthetic replay, use that model/run's calibrated speculative acceptance, not acceptance from synthetic text.
5. Plot total and output-only tokens/s/GPU against P90 interactivity; include TTFT p50/p90/p99 and errors.
   Keep decode-only interactivity separate from E2E-normalized interactivity.
6. Profile a low-concurrency and a throughput-bound point. Optimize the dominant cost and repeat the same workload.

For a controlled K3/K3.1 comparison, hold hardware, traffic, precision, and serving settings fixed where supported.
Keep that comparison separate from reproducing each model's independently tuned published result.

Next: select the GB300 GPU allocation size and retrieve the K3.1 AgentX run configurations.

[k3-agentx]: https://vllm.ai/blog/2026-09-08-vllm-agentx
[k3-fused]: https://vllm.ai/blog/2026-10-16-kimi-k3-megakernel
[k31]: https://vllm.ai/blog/2026-11-04-kimi-k31-pro
[recipe-k3]: https://recipes.vllm.ai/moonshotai/Kimi-K3
[recipe-k31]: https://recipes.vllm.ai/moonshotai/Kimi-K3.1-Pro
[agentx]: https://inferencex.semianalysis.com/agentx/quickstart
