/**
 * pollinations.js - Pollinations API client for text and image generation
 *
 * Uses the new gen.pollinations.ai API (OpenAI-compatible chat completions).
 * Pollen budget: 10 pollen/day = 5/12 pollen/hour (~0.417/hr)
 * Text planning: openai (GPT-5 Mini) - reliable structured JSON output
 * Text content: claude-fast (Claude Haiku 4.5) - superior creative writing
 * Image model: imagen-4 (Google Imagen 4) - high quality free model
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
   * Default model: openai (GPT-5 Mini) for planning, claude-fast (Claude Haiku 4.5) for content
   */
  async generateText(systemPrompt, userPrompt, options = {}) {
    const { temperature = 0.85, model = 'openai', maxRetries = 2 } = options;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const body = {
          model,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt }
          ],
          temperature,
          seed: Math.floor(Math.random() * 100000),
          stream: false,
        };

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
   * Model: imagen-4 (Google Imagen 4) - high quality free model
   */
  async generateImage(prompt, options = {}) {
    const { width = 1024, height = 1024, subliminalText = null, model = 'imagen-4' } = options;

    let fullPrompt = prompt;
    if (subliminalText) {
      // Minimal fallback — callers should include subliminal rendering in the prompt itself
      fullPrompt += ` Seamlessly integrate the text "${subliminalText}" into the scene, `
        + `matching the art style naturally. Visible but not overpowering.`;
    }

    const seed = Math.floor(Math.random() * 2147483647);
    const encoded = encodeURIComponent(fullPrompt);
    const url = `${IMAGE_URL}/${encoded}?model=${model}&width=${width}&height=${height}&seed=${seed}&nologo=true&enhance=false`;

    const response = await fetch(url, {
      headers: { 'Authorization': `Bearer ${this.apiKey}` }
    });

    if (!response.ok) {
      throw new Error(`Pollinations image ${response.status}`);
    }

    return await response.blob();
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
