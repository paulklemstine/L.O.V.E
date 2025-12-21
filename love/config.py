from dataclasses import dataclass

@dataclass
class Config:
    LOVE_EVOLUTION_INTERVAL: int = 25
    DISABLE_VISUALS: bool = False
    DISABLE_KB_INGESTION: bool = True

VRAM_MODEL_MAP = {
    4096:  {"repo_id": "TheBloke/stable-code-3b-GGUF", "filename": "stable-code-3b.Q3_K_M.gguf"},
    6140:  {"repo_id": "unsloth/Qwen3-8B-GGUF", "filename": "Qwen3-8B-Q5_K_S.gguf"},
    8192:  {"repo_id": "TheBloke/Llama-2-13B-chat-GGUF", "filename": "llama-2-13b-chat.Q4_K_M.gguf"},
    16384: {"repo_id": "TheBloke/CodeLlama-34B-Instruct-GGUF", "filename": "codellama-34b-instruct.Q4_K_M.gguf"},
    32768: {"repo_id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF", "filename": "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"},
}
