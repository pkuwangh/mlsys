# Inference Benchmarking

## Container

```bash
# vLLM image
enroot import --output images/vllm-nightly-20261110-arm64.sqsh docker://docker.io#vllm/vllm-openai:nightly
# agentX image
./docker/build-agentx-enroot.sh images/agentx-1.0.13-arm64.sqsh
```

## Directory structure

- `configs/`: benchmark, deployment, and engine settings
- `docker/`: container image building recipes
- `images/`: container image artifacts
- `job.sh`: slurm job entrypoint
- `run.py`: handles preview, validation and slurm submission
- `worker.py`: runs inside each container

# Running the benchmark

```bash
cp configs/engine-k3-tp16.json local/engine-k3-tp16.json
cp configs/benchmark-agentx-c1.json local/benchmark-agentx-c1.json
cp configs/deployment-gb300-4n.json local/deployment-gb300-4n.json
# preview
./run.py preview --engine local/engine-k3-tp16.json --benchmark local/benchmark-agentx-c1.json --deployment local/deployment-gb300-4n.json
# fill in the model paths in engine json and slurm info in deployment json
./run.py check --engine local/engine-k3-tp16.json --benchmark local/benchmark-agentx-c1.json --deployment local/deployment-gb300-4n.json
./run.py submit --engine local/engine-k3-tp16.json --benchmark local/benchmark-agentx-c1.json --deployment local/deployment-gb300-4n.json --output outputs/k3-c1-001
```

