
// Custom VLLM Provider Extension for Pi Agent
// This extension registers a local VLLM server as a provider.
// Dynamically fetches model config from the vLLM server.

export default async function (pi) {
    const VLLM_BASE_URL = "http://127.0.0.1:8000/v1";

    // Fetch model info from vLLM server
    let modelId = "Qwen/Qwen2.5-1.5B-Instruct"; // Default fallback
    let contextWindow = 4096;
    let maxTokens = 2048;

    try {
        const response = await fetch(`${VLLM_BASE_URL}/models`);
        if (response.ok) {
            const data = await response.json();
            if (data.data && data.data.length > 0) {
                const model = data.data[0];
                modelId = model.id;

                // vLLM returns max_model_len in model info
                if (model.max_model_len) {
                    contextWindow = model.max_model_len;
                    // Leave room for input by using ~60% of context for output
                    maxTokens = Math.floor(contextWindow * 0.6);
                }

                console.log(`[vLLM Extension] Loaded model: ${modelId}, context: ${contextWindow}, maxTokens: ${maxTokens}`);
            }
        }
    } catch (error) {
        console.warn(`[vLLM Extension] Failed to fetch model info, using defaults: ${error}`);
    }

    pi.registerProvider("vllm", {
        baseUrl: VLLM_BASE_URL,
        apiKey: "sk-test-key", // VLLM typically ignores this
        api: "openai-completions",
        models: [
            {
                id: modelId,
                name: `Local VLLM (${modelId.split('/').pop()})`,
                reasoning: false,
                input: ["text"],
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                contextWindow: contextWindow,
                maxTokens: maxTokens
            }
        ]
    });
}
