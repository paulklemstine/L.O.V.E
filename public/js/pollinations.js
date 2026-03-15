/**
 * pollinations.js - Pollinations API client for text and image generation
 *
 * Uses the new gen.pollinations.ai API (OpenAI-compatible chat completions).
 * Pollen budget: 10 pollen/day, 1 pollen/hr rate limit
 * Text planning: openai (GPT-5 Mini) - reliable structured JSON output
 * Text content: claude-fast (Claude Haiku 4.5) - best working creative model
 * Image model: dirtberry-pro - high realism, complex scenes
 */

const BASE_URL = 'https://gen.pollinations.ai';
const TEXT_URL = `${BASE_URL}/v1/chat/completions`;
const IMAGE_URL = `${BASE_URL}/image`;

export class PollinationsClient {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.callLog = [];
  }

  resetCallLog() { this.callLog = []; }
  getCallLog() { return this.callLog; }

  /**
   * Fetch live account stats from the Pollinations API.
   * Calls /account/balance, /account/profile, and /account/key in parallel.
   * Returns { balance, tier, nextResetAt, keyType, rateLimited, error? }.
   */
  async fetchAccountStats() {
    const headers = { 'Authorization': `Bearer ${this.apiKey}` };
    const results = { balance: null, tier: null, nextResetAt: null, keyType: null, rateLimited: null };

    try {
      const [balRes, profRes, keyRes] = await Promise.allSettled([
        fetch(`${BASE_URL}/account/balance`, { headers }),
        fetch(`${BASE_URL}/account/profile`, { headers }),
        fetch(`${BASE_URL}/account/key`, { headers }),
      ]);

      if (balRes.status === 'fulfilled' && balRes.value.ok) {
        const data = await balRes.value.json();
        results.balance = data.balance;
      }

      if (profRes.status === 'fulfilled' && profRes.value.ok) {
        const data = await profRes.value.json();
        results.tier = data.tier;
        results.nextResetAt = data.nextResetAt;
      }

      if (keyRes.status === 'fulfilled' && keyRes.value.ok) {
        const data = await keyRes.value.json();
        results.keyType = data.type;
        results.rateLimited = data.rateLimitEnabled;
      }
    } catch (err) {
      results.error = err.message;
    }

    return results;
  }

  /**
   * Generate text using Pollinations OpenAI-compatible API.
   * POST /v1/chat/completions — returns OpenAI JSON format.
   * Default model: openai (GPT-5 Mini) for planning, claude-airforce (Claude Sonnet 4.6) for content
   */
  async generateText(systemPrompt, userPrompt, options = {}) {
    const { temperature = 0.85, model = 'openai', maxRetries = 2,
      frequencyPenalty = 0.4, presencePenalty = 0.3 } = options;

    // Only claude models support penalty params on Pollinations
    const penaltiesSupported = model.startsWith('claude');

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const body = {
          model,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt }
          ],
          temperature,
          seed: Math.floor(Math.random() * 2147483647),
          stream: false,
        };

        if (penaltiesSupported) {
          body.frequency_penalty = frequencyPenalty;
          body.presence_penalty = presencePenalty;
        }

        // Force JSON output if prompt asks for JSON
        if (userPrompt.includes('Return ONLY valid JSON') || userPrompt.includes('Return ONLY raw JSON')) {
          body.response_format = { type: 'json_object' };
        }

        const response = await fetch(TEXT_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.apiKey}`
          },
          body: JSON.stringify(body)
        });

        if (!response.ok) {
          const errText = await response.text();
          throw new Error(`Pollinations text ${response.status}: ${errText.slice(0, 200)}`);
        }

        // New API returns OpenAI-compatible JSON
        const data = await response.json();
        const text = data.choices?.[0]?.message?.content || '';

        // Log prompt + response for activity log expansion
        this.callLog.push({
          label: options.label || 'LLM Call',
          systemPrompt,
          userPrompt,
          response: text,
          model,
        });

        return text;
      } catch (err) {
        if (attempt === maxRetries) throw err;
        // Exponential backoff (longer for pk_ key rate limits)
        const delay = 5000 * Math.pow(2, attempt);
        await new Promise(r => setTimeout(r, delay));
      }
    }
  }

  /**
   * Generate an image and return it as a Blob.
   * GET /image/{prompt} — returns binary image.
   * Model: flux-2-dev (FLUX.2 Dev) with fallback to flux (FLUX Schnell)
   */
  async generateImage(prompt, options = {}) {
    const { width = 1024, height = 1024, subliminalText = null, model = 'flux-2-dev', negativePrompt = null } = options;

    let fullPrompt = prompt;
    if (subliminalText) {
      fullPrompt += ` Seamlessly integrate the text "${subliminalText}" into the scene, `
        + `matching the art style naturally. Visible but not overpowering.`;
    }

    const seed = Math.floor(Math.random() * 2147483647);
    const encoded = encodeURIComponent(fullPrompt);

    const buildUrl = (m) => {
      let url = `${IMAGE_URL}/${encoded}?model=${m}&width=${width}&height=${height}&seed=${seed}&nologo=true&enhance=false`;
      if (negativePrompt) url += `&negative=${encodeURIComponent(negativePrompt)}`;
      return url;
    };

    // Try FLUX.2 Dev first, fall back to FLUX Schnell
    const models = [model, 'flux'];
    for (const m of models) {
      const response = await fetch(buildUrl(m), {
        headers: { 'Authorization': `Bearer ${this.apiKey}` }
      });

      if (response.ok) {
        return await response.blob();
      }

      // If last model, throw
      if (m === models[models.length - 1]) {
        throw new Error(`Pollinations image ${response.status}`);
      }
      // Otherwise fall back silently
      console.log(`[Pollinations] ${m} returned ${response.status}, falling back to next model`);
    }
  }

  /**
   * Generate text and parse JSON from the response.
   */
  async generateJSON(systemPrompt, userPrompt, options = {}) {
    const raw = await this.generateText(systemPrompt, userPrompt, {
      ...options,
      temperature: options.temperature || 0.7
    });
    return this.extractJSON(raw);
  }

  /**
   * Extract JSON from LLM response text.
   */
  extractJSON(text) {
    if (!text) return null;
    text = text.trim();

    // Strip markdown code blocks
    const codeBlockMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    if (codeBlockMatch) text = codeBlockMatch[1].trim();

    // Try direct parse
    try { return JSON.parse(text); } catch {}

    // Try to find JSON object
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      try { return JSON.parse(jsonMatch[0]); } catch {}
    }

    return null;
  }
}
