
// Custom VLLM Provider Extension for Pi Agent
// This extension registers a local VLLM server as a provider.

export default function (pi) {
    pi.registerProvider("vllm", {
        baseUrl: "http://127.0.0.1:8000/v1",
        apiKey: "sk-test-key", // VLLM typically ignores this or requires a placeholder
        api: "openai-completions",
        models: [
            {
                id: "Qwen/Qwen2.5-1.5B-Instruct", // Must match the loaded model in VLLM
                name: "Local VLLM (Qwen)",
                reasoning: false,
                input: ["text"],
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                contextWindow: 32768,
                maxTokens: 4096
            }
        ]
    });
}
