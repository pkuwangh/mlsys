#!/usr/bin/env python3

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
    logger.info(f"Loading model from: {model_path}")
    return LLM(model=model_path)


@nvtx.annotate("run")
def run(llm: LLM, prompts: list[str]) -> list[RequestOutput]:
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    logger.info(f"sampling_params: {sampling_params}")

    # concept
    # - engine_core: EngineCoreClient
    #   - model_executor: UniProcExecutor
    #     - driver_worker: WorkerWrapperBase
    #       - worker: gpu_worker.Worker
    #         - model_runner: GPUModelRunner
    #           - model: LlamaForCausalLM
    #           - kv_caches: list[Tensor]
    #     - collective_rpc to execute a RPC call on the worker; in constructor
    #       - init_worker
    #       - init_device
    #       - load_model
    #   - _initialize_kv_caches
    #     - model_executor.initialize_from_config -> worker.initialize_from_config
    #     - model_executor.compile_on_warm_up_model -> worker.compile_on_warm_up_model
    #       - model_runner.capture_model

    # generate call trace
    # - llm._validate_and_add_requests
    #   - llm_engine.add_request one by one
    #     - processor.process_input
    #     - engine_core.add_request
    #       - if sampling_params.n > 1 for parallel sampling, call add_request n times
    # - llm._run_engine
    #   while llm_engine.has_unfinished_requests:
    #     - step_outputs = llm_engine.step()  # one decoding iteration
    #       - EngineCore.step
    #         - scheduler.schedule; also token blocks to be swapped in/out
    #         - module_executor.execute_model
    #         - scheduler_context.append_output
    #         - process module output
    logger.info("Generating text...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    return outputs


if __name__ == "__main__":
    # init cuda context; otherwise nsys gets nothing!
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")

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
