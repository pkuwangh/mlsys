#!/usr/bin/env python3

import sglang as sgl
from utils import get_model_path


def main():
    model_path = get_model_path("Qwen/Qwen3.8-27B")
    llm = sgl.Engine(model_path=model_path)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    main()
