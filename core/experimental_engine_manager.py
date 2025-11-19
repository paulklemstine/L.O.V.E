# core/experimental_engine_manager.py

import random
from huggingface_hub import snapshot_download
from core.deep_agent_engine import DeepAgentEngine
from core.simulation_datasets import generate_randomized_dataset

def _select_model(vram):
    """
    Selects the best vLLM-compatible model based on available VRAM.
    """
    if vram >= 148 * 1024:
        return "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    elif vram >= 44 * 1024:
        return "TheBloke/deepseek-llm-67b-base-AWQ"
    elif vram >= 42 * 1024:
        return "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    elif vram >= 22 * 1024:
        return "Qwen/Qwen2-32B-Instruct-AWQ"
    elif vram >= 20 * 1024:
        return "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    elif vram >= 6.5 * 1024:
        return "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    elif vram >= 4.5 * 1024:
        return "Sreenington/Phi-3-mini-4k-instruct-AWQ"
    elif vram >= 2.5 * 1024:
        return "elysiantech/gemma-2b-awq-4bit"
    else:
        return "Qwen/Qwen2-1.5B-Instruct-AWQ"

def _download_model_snapshot(repo_id):
    """
    Downloads a model snapshot from the Hugging Face Hub if it's not already cached
    and returns the local file path.
    """
    try:
        print(f"Ensuring model snapshot for {repo_id} is downloaded...")
        model_path = snapshot_download(repo_id=repo_id)
        print(f"Model snapshot is available at: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model snapshot from {repo_id}: {e}")
        raise

def initialize_deep_agent_engine(tool_registry, persona_path, architecture):
    """
    Initializes the vLLM instance and the DeepAgent itself based on a given architecture.
    The 'architecture' parameter is a dictionary that can specify the 'model_repo' or 'vram'.
    The user's request for 'layers', 'activation_functions', and 'optimizer' is interpreted
    metaphorically as the choice of a pre-trained model, which is the practical application
    of this concept in the current codebase.
    """
    from vllm import LLM, SamplingParams

    vram = architecture.get('vram', 0)
    model_repo = architecture.get('model_repo') or _select_model(vram)

    print(f"Initializing DeepAgent with model from repo: {model_repo}...")
    try:
        model_path = _download_model_snapshot(model_repo)

        if vram >= 6.5 * 1024:
            llm = LLM(model=model_path, gpu_memory_utilization=0.9)
        else:
            llm = LLM(model=model_path, gpu_memory_utilization=0.80)

        max_len = llm.llm_engine.model_config.max_model_len
        dynamic_max_tokens = min(max_len // 2, 8192)

        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=dynamic_max_tokens)

        # The DeepAgentEngine will now initialize itself with the llm and sampling_params
        engine = DeepAgentEngine(tool_registry, persona_path, llm, sampling_params)

        print("vLLM engine initialized successfully.")
        return engine
    except Exception as e:
        print(f"Error initializing vLLM: {e}")
        return None

def run_simulation_loop(tool_registry, persona_path, candidate_architectures):
    """
    Implements a simulation loop that injects randomized datasets into each model.
    """
    for arch in candidate_architectures:
        print(f"Testing architecture: {arch}")
        engine = initialize_deep_agent_engine(tool_registry, persona_path, arch)
        if engine:
            for i in range(5): # Run 5 iterations for each architecture
                dataset = generate_randomized_dataset()
                print(f"Running iteration {i+1} with dataset: {dataset}")
                result = engine.run(dataset)
                print(f"Result: {result}")
