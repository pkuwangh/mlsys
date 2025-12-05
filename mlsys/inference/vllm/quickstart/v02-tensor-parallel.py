#!/usr/bin/env python3

import os
import nvtx
import torch
from loguru import logger
from torch.cuda import cudart
from utils import get_model_path
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput


@nvtx.annotate("setup")
def setup() -> LLM:
    model_path = get_model_path("NousResearch/Hermes-3-Llama-3.1-8B/")
    logger.warning(f"[pid={os.getpid()}] Loading model from: {model_path}")
    return LLM(
        model=model_path,
        tensor_parallel_size=2,
        compilation_config={"cudagraph_mode": "piecewise"},
    )


@nvtx.annotate("run")
def run(llm: LLM, prompts: list[str]) -> list[RequestOutput]:
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16)
    logger.info(f"sampling_params: {sampling_params}")

    logger.info("Generating text...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    return outputs


if __name__ == "__main__":
    # with TP=2, the worker will init the cuda context?
    # torch.cuda.set_device(0)
    # torch.zeros(1, device="cuda")

    llm = setup()

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompts = prompts
    logger.info(f"Prepared {len(prompts)} prompts for generation.")

    cudart().cudaProfilerStart()
    outputs = run(llm, prompts)
    torch.cuda.synchronize()
    cudart().cudaProfilerStop()

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        logger.info(f"OSL={len(output.outputs[0].token_ids)}: {prompt} -> {generated_text}")
