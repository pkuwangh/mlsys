#!/usr/bin/env python3

import argparse

def main(args):
    # lazy import so the CLI help message can show up quickly
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    print("=============================================")
    print(model)
    print("=============================================")

    if not args.run_inference:
        return
    if args.gpu:
        model.to("cuda")

    prompt = "Michael Burry is"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)
    print("Inputs:")
    for k, v in inputs.items():
        print(f"\t{k}: {v.shape}")
    print(f"Model is on {model.device}")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=160)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to the model")
    parser.add_argument("--run-inference", "-r", action="store_true", help="Run an inference")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
