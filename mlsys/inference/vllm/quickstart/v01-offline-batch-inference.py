#!/usr/bin/env python3

from loguru import logger
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

logger.info(f"sampling_params: {sampling_params}")

llm = LLM(model="gpt2")

logger.info("Generating text...")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    logger.info(f"{prompt!r} -> {generated_text!r}")
