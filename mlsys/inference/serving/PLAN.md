# Kimi K3 / K3.1 Serving Study

Models: `moonshotai/Kimi-K3` and `moonshotai/Kimi-K3.1-Pro`.
Hardware: MIS CDG GB300 cluster, four GPUs per Slurm node. Start with K3.
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
2. Reproduce K3 first; defer K3.1. Use the published TP/DCP/EP, KV dtype/offload, batch limits,
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

## K3 Reproduction Preparation — November 9, 2026

Slurm reports `mis-cdg-slurm-1`, partitions `batch` (4-hour limit) and
`batch_long`, with `gpu:4` and `nvidia_gb300,mnnvl` features. Set the Slurm
account through `SBATCH_ACCOUNT` in the shell. Idle nodes were
available when inspected; availability is not a reservation. No benchmark
job has been submitted for this study.

### First point: AgentX concurrency 1

Use one aggregated TP16/DCP16 engine on four GB300 trays, specialized decode,
and the full AgentX corpus. The runnable template is
`configs/engine-k3-tp16.json`, `configs/benchmark-agentx-c1.json`, and
`configs/deployment-gb300-4n.json`;
commands and container/patch workflows are in
`NOTES.md`. Checkpoint paths will be supplied by the user. Image paths also
remain unset until compatible ARM64 images are prepared and tested.

The engine template uses the current preset's DSpark K5 and the K3 probabilistic/block
calibration AL=3.62 for synthetic replay. Record this as a new baseline, not an
exact historical reproduction. Pin image and harness, perform a short smoke
run, then run standard warmup and a one-hour measurement at concurrency 1.
Follow with concurrency 2, 4 and 8 on the same GPU budget and configuration.

### Follow-up: 16-GPU decode comparison

Allocate four exclusive nodes in one NVLink domain. Verify driver, GPU type,
NVLink/IMEX connectivity and multicast on the allocated nodes before loading.
Run one aggregated TP16 engine, with stock and specialized decode using the
same pinned image and checkpoints. Start without speculation, then add DSpark.

| Stage | Workload | Published reference |
| --- | --- | --- |
| No speculation | ISL 1, OSL 1024, greedy; batches 1, 2, 4, 8 | TP16 batch-1 decode step about 3.90 ms specialized vs. 7.38 ms stock |
| Controlled speculation | TP16, DSpark 7 draft tokens, synthetic acceptance length 6, batch 1 | Blog: 842 decode tokens/s; PR table: 838 vs. 489 stock |
| Context sensitivity | ISL 32K, DCP16, DSpark 7, synthetic acceptance 3.78 | Separate long-context comparison; exclude prefill from decode-only measurements |

The [optimization PR][k3-pr] records the benchmark definitions. Preserve the
small blog/PR numerical discrepancy in reporting. The short-context headline's
complete launch configuration, including DCP choice, still needs its original
run artifacts; do not silently substitute the long-context recipe.

Use FP8 KV and FP32 recurrent SSM state. The [current low-latency recipe][k3-low]
uses BF16 convolution state, V2 model runner, full decode CUDA graphs,
PP1 and no expert parallelism. Enable the optimized path with
`--additional-config '{"decode_backend":"latency_specialized"}'` and confirm
it is active in logs/profiles. Avoid ReplaySSM, eager mode and unsupported
batch shapes for this comparison. Repeat each selected point three times.

The current recipe is a starting point, not the historical benchmark command:
its TP16 preset uses DCP16, **five** draft tokens and real block rejection,
whereas the headline uses **seven** draft tokens and forced acceptance.
Rebuild graph capture sizes for the actual draft/verification shapes. Measure
decode steps with profiling; streamed chunks (`stream-interval=10` in the
recipe) are not individual token timings. Validate real generation separately
with forced acceptance disabled.

### Historical AgentX comparison

The September target remains 16 GPUs / concurrency 48 / 11.8K total tokens/s/GPU
and 62.7 P90 tokens/s/user. The blog links to [run 441066][k3-agentx-run], but its
rendered page did not expose the original configuration during this inspection.
Its image, topology, acceptance distribution, harness revision, warmup and
measurement duration remain unresolved.

The [current GB300 AgentX preset][k3-pd] instead defaults to **1P2D, 24 GPUs,
concurrency 24**. Each worker uses TP8/DCP8 across two trays. It cites
InferenceX PR 4638, lane revision `bc4d49f5df30`, and image digest
`sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3`.
This is a separate modern reproduction target, not an exact September replay.

Recover the chosen run's complete source configuration rather than relying
only on generated commands: the current recipe says worker replication,
Mooncake sizing and frontend session affinity need additional settings.
Use the full corpus, preserve context lengths, and record both ITL-based and
E2E-normalized interactivity with explicit formulas. Use the selected run's
K3 acceptance distribution; AL=6 is only the controlled headline experiment.

### Local assets and remaining gates

The shared Hugging Face cache contains snapshot directories for:

- K3: `f831ab66814297da540d832a5235f8e904f29d06`.
- Inferact K3 DSpark: `cf6b8244620e7ea4b0651d214f28e89eac75bed6`.

Keep the site-specific shared cache root in an ignored `*.local.json` config.
Directory presence is verified; shard completeness and container readability
still need checking. Preserve the shared blob symlinks when mounting snapshots.

The neighboring `imaginaire4-prod-1` repository contains Slurm launchers,
cached images and prior K3/K3.1 experiments. Its deployed K3 configuration
uses a different image digest (`aa018ae0…`), a 262144 context cap, and ReplaySSM;
it is not the reproduction configuration above. Existing services should not
receive benchmark traffic or be reconfigured as part of this study.

Next: prepare ARM64 server and benchmark images, fill the user-supplied checkpoint
paths in a local config, validate mounts/flags with a GPU smoke run, and measure
the isolated four-node AgentX concurrency-1 point. Historical reproduction
claims remain gated on recovering the original run manifests.

[k3-agentx]: https://vllm.ai/blog/2026-09-08-vllm-agentx
[k3-fused]: https://vllm.ai/blog/2026-10-16-kimi-k3-megakernel
[k31]: https://vllm.ai/blog/2026-11-04-kimi-k31-pro
[recipe-k3]: https://recipes.vllm.ai/moonshotai/Kimi-K3
[recipe-k31]: https://recipes.vllm.ai/moonshotai/Kimi-K3.1-Pro
[agentx]: https://inferencex.semianalysis.com/agentx/quickstart
[k3-pr]: https://github.com/vllm-project/vllm/pull/61867
[k3-low]: https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=gb300&strategy=multi_node_tp&preset=gb300_low_latency
[k3-pd]: https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=gb300&strategy=pd_cluster&preset=gb300_agentx_pd
[k3-agentx-run]: https://inferencex.semianalysis.com/inference/agentic/441066
