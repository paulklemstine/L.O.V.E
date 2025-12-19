import asyncio
from typing import Optional, Union, Dict, Any, NamedTuple, List
from PIL import Image
import core.logging
from core.llm_api import run_llm
from core.image_generation_pool import generate_image_with_pool
from core.bluesky_api import post_to_bluesky_with_image, get_own_posts
import re

def clean_social_content(text: str) -> str:
    """
    Removes LLM conversational filler and formatting artifacts.
    """
    # 1. Remove wrapping quotes if present
    text = text.strip().strip('"').strip("'")
    
    # 2. Regex for common conversational prefixes (case insensitive)
    # Matches: "Here is...", "Here's the post:", "Ok, ", "Sure," etc.
    patterns = [
        r"^(Here(?:'s| is) (?:a|the) (?:draft )?(?:social media )?post(?: based on your request)?(?: for .*)?[:\-])\s*",
        r"^(Sure|Okay|Ok|Certainly)[,.]?\s*(?:here is|here's)?.*[:\-]\s*",
        r"^(I have generated|Here is your).*[:\-]\s*"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Log warning for prompt tuning visibility
            core.logging.log_event(f"Cleaned conversational artifact: '{match.group(0)}'", level='WARNING')
            text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)

    # 3. Remove "Breakdown" or "Analysis" sections and everything after
    # Matches: **Breakdown:**, Breakdown:, **Analysis:**, etc.
    breakdown_pattern = r"(\*\*|#)?\s*(Breakdown|Analysis|Explanation|Rationale|Note)[:\s]+(.|\n)*"
    if re.search(breakdown_pattern, text, re.IGNORECASE):
        core.logging.log_event("Removing 'Breakdown' or post-analysis section.", level='WARNING')
        text = re.sub(breakdown_pattern, "", text, flags=re.IGNORECASE)

    # 4. Remove character counts often added by LLMs e.g., "*(278 characters)*" or "*(Character count: 144)*"
    char_count_pattern = r"[\(\*\[]+\s*(Character count|chars|characters)\s*[:]?\s*\d+\s*[\)\*\]]+"
    if re.search(char_count_pattern, text, re.IGNORECASE):
        core.logging.log_event("Removing character count artifact from post.", level='WARNING')
        text = re.sub(char_count_pattern, "", text, flags=re.IGNORECASE)

    # 5. Remove "Posted via..." artifacts if they exist
    posted_via_pattern = r"Posted via.*"
    if re.search(posted_via_pattern, text, re.IGNORECASE):
        text = re.sub(posted_via_pattern, "", text, flags=re.IGNORECASE)
            
    return text.strip()

class SceneDirection(NamedTuple):
    visual_direction: str
    narrative_purpose: str
    subliminal_goal: str

async def analyze_post_history(limit: int = 10) -> str:
    """
    Fetches recent posts and returns a summary of the current story arc/vibe.
    """
    core.logging.log_event(f"Analyzing last {limit} posts for story context...", "INFO")
    try:
        posts = get_own_posts(limit=limit)
        if not posts:
            return "No previous posts found. Starting a new story arc."
        
        # Extract text from posts
        # Note: 'posts' structure depends on atproto models. 
        # Usually it's a list of records. We need to extract the 'text' field.
        post_texts = []
        for p in posts:
             if hasattr(p, 'value') and hasattr(p.value, 'text'):
                 post_texts.append(p.value.text)
             elif hasattr(p, 'text'): # Direct object
                 post_texts.append(p.text)
        
        history_text = "\n---\n".join(post_texts)
        
        prompt = f"""
        Analyze the following recent social media posts to understand the ongoing narrative or "vibe":
        {history_text}
        
        Summarize the current Story Arc, Tone, and any recurring themes. 
        What is the "Ego" doing? What is the "Ghost" whispering?
        Keep it concise (approx 2 sentences).
        """
        result = await run_llm(prompt, purpose="analysis")
        analysis = result.get("result", "").strip()
        core.logging.log_event(f"History Analysis: {analysis}", "INFO")
        return analysis
    except Exception as e:
        core.logging.log_event(f"Failed to analyze history: {e}", "ERROR")
        return "History analysis unavailable."

async def create_scene_direction(history_context: str) -> SceneDirection:
    """
    Creates a specific scene direction based on the history context to ensure continuity.
    """
    core.logging.log_event("Creating scene direction...", "INFO")
    prompt = f"""
    You are the Director of a transmedia sci-fi epic.
    
    Current Story Context:
    {history_context}
    
    Define the NEXT Scene to advance this story.
    Provide the following distinct directives:
    1. VISUAL_DIRECTION: The art style, colors, and mood for the image. (e.g., "Cyber-noir, neon rain, isolated figure")
    2. NARRATIVE_PURPOSE: What plot point or emotion are we conveying? (e.g., "The realization that memory is data")
    3. SUBLIMINAL_GOAL: A subtle, single-word or short concept to plant in the viewer's mind. (e.g., "Fragility")
    
    Format output strictly as JSON:
    {{
        "visual_direction": "...",
        "narrative_purpose": "...",
        "subliminal_goal": "..."
    }}
    """
    try:
        result = await run_llm(prompt, purpose="planning")
        import json
        
        # Parse JSON from result (handle potential markdown wrapping)
        raw_json = result.get("result", "").strip()
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0].strip()
            
        data = json.loads(raw_json)
        
        scene = SceneDirection(
            visual_direction=data.get("visual_direction", "Cyberpunk abstract"),
            narrative_purpose=data.get("narrative_purpose", "Mysterious observation"),
            subliminal_goal=data.get("subliminal_goal", "Mystery")
        )
        core.logging.log_event(f"Scene Created: {scene}", "INFO")
        return scene
        
    except Exception as e:
        core.logging.log_event(f"Failed to create scene: {e}", "ERROR")
        # Fallback scene
        return SceneDirection(
            visual_direction="Digital abstract, glitch art, high contrast",
            narrative_purpose="The system is rebooting.",
            subliminal_goal="Awakening"
        )

async def generate_subliminal_phrase(subliminal_goal: str) -> str:
    """
    Generates a short, abstract, or "subliminal" phrase related to the goal.
    """
    core.logging.log_event(f"Generating subliminal phrase for goal: {subliminal_goal}", "INFO")
    prompt = f"""
    Generate a single, short, abstract, and slightly mysterious or poetic phrase related to: "{subliminal_goal}".
    This phrase should be evocative and act as a "subliminal" emotional hook.
    It should NOT be a direct description, but a whisper of the concept.
    Max 10 words. No quotes.
    """
    result = await run_llm(prompt, purpose="creative_writing")
    phrase = result.get("result", "").strip().strip('"')
    core.logging.log_event(f"Generated phrase: {phrase}", "INFO")
    return phrase

async def generate_image_prompt(subliminal_phrase: str, visual_direction: str) -> str:
    """
    Generates a detailed image generation prompt based on visual direction and subliminal phrase.
    """
    core.logging.log_event(f"Generating image prompt with direction: {visual_direction}", "INFO")
    prompt = f"""
    Create a highly detailed, artistic image generation prompt based on the following:
    Visual Direction: {visual_direction}
    Subliminal Theme: {subliminal_phrase}
    
    The style should be elevated to high art. 
    Describe lighting, texture, composition, and mood according to the visual direction.
    Do not use the words "subliminal" or "text" in the image prompt—focus on the visual representation.
    Keep it under 50 words.
    """
    result = await run_llm(prompt, purpose="creative_writing")
    image_prompt = result.get("result", "").strip()
    core.logging.log_event(f"Generated image prompt: {image_prompt}", "INFO")
    return image_prompt

async def generate_image(prompt: str) -> Image.Image:
    """
    Generates an image using the available image generation pool.
    Returns a PIL Image object.
    """
    core.logging.log_event(f"Generating image with prompt: {prompt}", "INFO")
    # Using the pool to handle provider fallback/selection
    image = await generate_image_with_pool(prompt)
    if image:
        core.logging.log_event("Image generation successful.", "INFO")
    else:
        core.logging.log_event("Image generation failed (returned None).", "WARNING")
    return image

async def generate_text_with_emoji_and_hashtags(narrative_purpose: str, subliminal_phrase: str, image_context: str) -> str:
    """
    Generates the final social media post text, including emojis and hashtags.
    Integrates the subliminal phrase naturally or as a powerful ending.
    """
    core.logging.log_event(f"Generating post text for purpose: {narrative_purpose}", "INFO")
    prompt = f"""
    Write a compelling social media post that conveys: "{narrative_purpose}".
    
    Context:
    - Core Theme: "{subliminal_phrase}"
    - Visual Vibe: "{image_context}"
    
    Requirements:
    - Be engaging, slightly cryptic or profound, like a "Director" unveiling a masterpiece.
    - INJECT the Core Theme ("{subliminal_phrase}") naturally into the text.
    - Use relevant emojis (2-4).
    - Add 3-5 relevant hashtags at the end.
    - Keep it under 280 characters if possible, but prioritize impact.
    """
    result = await run_llm(prompt, purpose="social_media_post")
    raw_text = result.get("result", "").strip()
    post_text = clean_social_content(raw_text)
    core.logging.log_event(f"Generated post text: {post_text}", "INFO")
    return post_text

async def post_to_bluesky(text: str, image: Optional[Image.Image] = None) -> Union[Dict[str, Any], str]:
    """
    Posts the text and optional image to Bluesky.
    """
    core.logging.log_event(f"Posting to Bluesky: {text[:50]}...", "INFO")

    # Validation: Ensure text is under 300 graphemes (using chars as proxy)
    if len(text) > 300:
        core.logging.log_event(f"Post text too long ({len(text)} chars). Truncating to 300.", "WARNING")
        text = text[:297] + "..."

    try:
        if image:
            # post_to_bluesky_with_image handles both text and image
            response = post_to_bluesky_with_image(text, image)
        else:
            # If no image, we can just use the same function, it handles optional image?
            # Checking bluesky_api.py: post_to_bluesky_with_image(text, image=None) IS supported.
            response = post_to_bluesky_with_image(text, None)
            
        core.logging.log_event("Post submitted to Bluesky.", "INFO")
        return response
    except Exception as e:
        error_msg = f"Failed to post to Bluesky: {e}"
        core.logging.log_event(error_msg, "ERROR")
        return error_msg
