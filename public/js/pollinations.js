/**
 * pollinations.js - Pollinations API client for text and image generation
 *
 * Pollen budget: 10 pollen/day = 5/12 pollen/hour (~0.417/hr)
 * Text model: openai (GPT-5 Mini) - 950 responses/pollen, best non-paid quality
 * Image model: gptimage (GPT Image 1 Mini) - 80 images/pollen, renders text in-scene
 * Estimated per hour (~12 cycles): ~0.17 pollen (within 0.417/hr budget)
 */

const TEXT_URL = 'https://text.pollinations.ai/';
const IMAGE_URL = 'https://image.pollinations.ai/prompt';

// Pollen tracking
let pollenUsed = 0;
let pollenResetTime = Date.now() + 3600000;

export class PollinationsClient {
  constructor(apiKey) {
    this.apiKey = apiKey;
  }

  /**
   * Get pollen usage stats.
   */
  getPollenStats() {
    if (Date.now() > pollenResetTime) {
      pollenUsed = 0;
      pollenResetTime = Date.now() + 3600000;
    }
    return {
      used: pollenUsed.toFixed(4),
      remaining: Math.max(0, 1 - pollenUsed).toFixed(4),
      resetsIn: Math.max(0, Math.floor((pollenResetTime - Date.now()) / 60000))
    };
  }

  /**
   * Generate text using Pollinations API.
   * Uses POST with messages array.
   * Model: openai (GPT-5 Mini) - best non-paid quality
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
          jsonMode: false,
          private: true
        };

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

        // Base endpoint returns plain text
        const text = await response.text();

        // Track pollen (~0.001 per call with openai model)
        pollenUsed += 0.001;

        return text;
      } catch (err) {
        if (attempt === maxRetries) throw err;
        // Exponential backoff
        await new Promise(r => setTimeout(r, 3000 * (attempt + 1)));
      }
    }
  }

  /**
   * Generate an image and return it as a Blob.
   * Model: gptimage (GPT Image 1 Mini) - best quality, excellent text rendering
   */
  async generateImage(prompt, options = {}) {
    const { width = 1024, height = 1024, subliminalText = null, model = 'gptimage' } = options;

    let fullPrompt = prompt;
    if (subliminalText) {
      fullPrompt += ` Seamlessly integrate the text "${subliminalText}" into the scene - `
        + `rendered in a style that fits naturally (glowing neon, flickering fire, crystalline ice, `
        + `liquid chrome, ethereal light, smoky wisps, or electric plasma). `
        + `Place it where it feels right within the composition. Emphasized but not overpowering.`;
    }

    const seed = Math.floor(Math.random() * 2147483647);
    const encoded = encodeURIComponent(fullPrompt);
    const url = `${IMAGE_URL}/${encoded}?model=${model}&width=${width}&height=${height}&seed=${seed}&enhance=true&nologo=true&private=true`;

    const response = await fetch(url, {
      headers: { 'Authorization': `Bearer ${this.apiKey}` }
    });

    if (!response.ok) {
      throw new Error(`Pollinations image ${response.status}`);
    }

    // Track pollen (~0.0125 per image for gptimage)
    pollenUsed += 0.0125;

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
