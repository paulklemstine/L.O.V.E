
// Custom VLLM Provider Extension for Pi Agent
// This extension registers a local VLLM server as a provider.

export default async function (pi) {
    const VLLM_BASE_URL = "http://127.0.0.1:8000/v1";

    console.log("[vLLM Extension] Initializing...");

    // Default values
    let modelId = "Qwen/Qwen2.5-1.5B-Instruct";
    let contextWindow = 16384;
    let maxTokens = 2048;

    // --- SES-Safe Config Loading (Best Effort) ---
    try {
        if (typeof process !== 'undefined' && process.env && process.env.VLLM_EXTENSION_CONFIG_PATH) {
            const fs = await import('fs');
            if (fs.existsSync(process.env.VLLM_EXTENSION_CONFIG_PATH)) {
                try {
                    const config = JSON.parse(fs.readFileSync(process.env.VLLM_EXTENSION_CONFIG_PATH, 'utf8'));
                    if (config.model_id) modelId = config.model_id;
                    // Use config values if available
                    if (config.context_window) contextWindow = config.context_window;
                    if (config.max_tokens) maxTokens = config.max_tokens;
                    console.log(`[vLLM Extension] Read config: ${JSON.stringify(config)}`);
                } catch (e) { /* ignore parse error */ }
            }
        }
    } catch (e) {
        console.warn(`[vLLM Extension] Config loading failed (expected in SES): ${e}`);
    }

    // --- API Discovery (Best Effort) ---
    try {
        if (typeof fetch !== 'undefined') {
            const response = await fetch(`${VLLM_BASE_URL}/models`);
            if (response.ok) {
                const data = await response.json();
                if (data.data && data.data.length > 0) {
                    const model = data.data[0];
                    modelId = model.id;
                    if (model.max_model_len) {
                        contextWindow = model.max_model_len;
                        maxTokens = Math.floor(contextWindow * 0.2);
                    }
                    console.log(`[vLLM Extension] Discovered model: ${modelId}, context=${contextWindow}`);
                }
            }
        }
    } catch (e) {
        console.warn(`[vLLM Extension] API discovery failed: ${e}`);
    }

    // --- Final Registration ---
    console.log(`[vLLM Extension] Registering vLLM provider: context=${contextWindow}, maxTokens=${maxTokens}`);

    pi.registerProvider("vllm", {
        baseUrl: VLLM_BASE_URL,
        apiKey: "sk-test-key",
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
