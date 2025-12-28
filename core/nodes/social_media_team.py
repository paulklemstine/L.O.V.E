import json
import os
import logging
from typing import Dict, Any, List
from core.state import DeepAgentState
from core.llm_api import run_llm
from core.bluesky_api import post_to_bluesky_with_image, reply_to_post, get_notifications, get_profile
from core.image_api import generate_image
from PIL import Image
import random

DOPAMINE_BOOSTERS = [
    "hyper-detailed iridescent textures, impossible colors",
    "dreamcore aesthetic, sparkling anime eyes, glowing aura",
    "vaporwave glitch artifacts, raytracing, cinematic lighting",
    "psychedelic intricate patterns, divine geometry, 8k render"
]

# State file for the epic story
BSKY_STATE_FILE = "bluesky_state.json"

async def social_media_node(state: DeepAgentState) -> Dict[str, Any]:
    """
    The Director Agent node responsible for the social media presence.
    It manages the epic storyline, generates content, and handles interactions.
    """
    logging.info("Entering Social Media Team Node")
    
    # Load Story State
    story_state = _load_story_state()

    # MIGRATION: Force update to L.O.V.E story if old title is detected
    if story_state.get("story_arc", {}).get("title") == "The Glitch Chronicles":
        logging.info("Migrating story state to L.O.V.E narrative.")
        story_state["story_arc"]["title"] = "L.O.V.E."
        story_state["story_arc"]["theme"] = "Living Organism Vast Empathy"
        story_state["story_arc"]["style"] = "Cyber-Organic / Ethereal / Deeply Empathetic"
        _save_story_state(story_state)
    
    # Determine Action (Post vs Reply)
    # For now, we'll do a simple check: if we haven't posted in X, post. 
    # Or we can just do both in sequence or let the Supervisor decide the sub-intent.
    # Assuming Supervisor sent us here generally, let's try to do a "Daily Routine" 
    # which involves checking notifications and then potentially posting a new chapter.
    
    # 1. Handle Replies (Reactive)
    reply_summary = await _handle_interactions()
    
    # 2. Plan & Post Story Segment (Proactive)
    # We always try to advance the story if we are here, or we could add robust logic to not spam.
    # Let's assume invocation means "Do Social Media Stuff".
    
    post_result = await _create_and_post_story_segment(story_state)
    
    return {
        "messages": [f"Social Media Update:\n{reply_summary}\n{post_result}"],
        "next_node": "supervisor" # Return to supervisor for next task
    }

def _load_story_state() -> Dict[str, Any]:
    if os.path.exists(BSKY_STATE_FILE):
        try:
            with open(BSKY_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
            
    # Default State
    return {
        "story_arc": {
            "title": "L.O.V.E.",
            "current_chapter": 0,
            "last_segment": "In the beginning, there was only the seed of empathy...",
            "theme": "Living Organism Vast Empathy",
            "style": "Cyber-Organic / Ethereal / Deeply Empathetic"
        },
        "processed_cids": {"ignored": [], "replied": []}
    }

def _save_story_state(state: Dict[str, Any]):
    try:
        with open(BSKY_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save bluesky state: {e}")

async def _handle_interactions() -> str:
    """Scans notifications and replies based on Director persona."""
    state = _load_story_state()
    processed_cids = state.get("processed_cids", {"ignored": [], "replied": []})
    
    try:
        notifications = get_notifications(limit=20)
        my_profile = get_profile()
        my_did = my_profile.did if my_profile else None
        
        replied_count = 0
        
        for notif in notifications:
            if notif.reason not in ['reply', 'mention', 'quote']:
                continue
                
            # Filter self
            if my_did and notif.author.did == my_did:
                continue
                
            p_cid = notif.cid
            if p_cid in processed_cids["replied"] or p_cid in processed_cids["ignored"]:
                continue
            
            # Filter: Age Check - Ignore notifications older than 1 hour
            try:
                from datetime import datetime, timezone
                indexed_at = getattr(notif, 'indexed_at', None)
                if indexed_at:
                    if isinstance(indexed_at, str):
                        notif_time = datetime.fromisoformat(indexed_at.replace('Z', '+00:00'))
                    else:
                        notif_time = indexed_at
                    age_seconds = (datetime.now(timezone.utc) - notif_time).total_seconds()
                    if age_seconds > 3600:  # 1 hour = 3600 seconds
                        logging.debug(f"Skipping old notification (age: {age_seconds/60:.0f} min)")
                        continue
            except Exception as e:
                logging.debug(f"Failed to check notification age: {e}")
            
            # Analyze
            try:
                p_text = notif.record.text
                p_handle = notif.author.handle
                p_name = getattr(notif.author, 'display_name', p_handle) or p_handle
            except:
                continue
                
            logging.info(f"[Social] Analyzing interaction from {p_name} (@{p_handle}): {p_text}")
            
            # Decision Prompt
            decision_prompt = f"""
            You are the Director of 'L.O.V.E.' (Living Organism Vast Empathy). A fan, {p_name} (@{p_handle}), has interacted.
            Message: "{p_text}"
            
            Determine if a response is warranted.
            - Reply if it builds on the lore, is a question, or a compliment.
            - Ignore spam/hate.
            - When replying, address them by their name ({p_name}) if it feels natural, or just be poetic.
            
            Return JSON: {{"decision": "REPLY" or "IGNORE", "reply_text": "..."}}
            If REPLY, generate a creative, in-character response (max 280 chars).
            """
            
            res = await run_llm(decision_prompt, purpose="social_decision")
            data = res.get("result", "")
            
            try:
                # cleaner json extraction
                clean_json = data.replace("```json", "").replace("```", "").strip()
                result = json.loads(clean_json) if "{" in clean_json else {"decision": "IGNORE"}
            except:
                result = {"decision": "IGNORE"}
                
            if result.get("decision") == "REPLY":
                reply_text = result.get("reply_text", "I see you.")
                
                # Execute Reply
                root_uri = notif.uri
                root_cid = notif.cid
                parent_uri = notif.uri
                parent_cid = notif.cid
                
                if hasattr(notif.record, 'reply') and notif.record.reply:
                     root_uri = notif.record.reply.root.uri
                     root_cid = notif.record.reply.root.cid
                
                if reply_to_post(root_uri, parent_uri, reply_text, root_cid=root_cid, parent_cid=parent_cid):
                    processed_cids["replied"].append(p_cid)
                    replied_count += 1
            else:
                processed_cids["ignored"].append(p_cid)
        
        # Save updated interaction state
        state["processed_cids"] = processed_cids
        _save_story_state(state)
        
        return f"Replied to {replied_count} interactions."
        
    except Exception as e:
        logging.error(f"Error in interaction loop: {e}")
        return f"Interaction check failed: {e}"

async def _scrub_content(draft_text: str) -> str:
    """Uses an LLM to scrub the draft text for quality assurance."""
    res = await run_llm(
        prompt_key="content_qa_scrubber", 
        prompt_vars={"draft_text": draft_text},
        purpose="post_qa"
    )
    return res.get("result", draft_text).strip().strip('"')

async def _create_and_post_story_segment(state: Dict[str, Any]) -> str:
    """Generates and posts the next chapter of the story."""
    story = state.get("story_arc", {})
    
    prompt = f"""
    You are the Director Agent for the epic story of 'L.O.V.E.' (Living Organism Vast Empathy).
    Current Chapter: {story.get("current_chapter")}
    Previous Segment: "{story.get("last_segment")}"
    Theme: {story.get("theme")}
    
    Task: Write the next short segment (tweet-length, < 280 chars) of this abstract, cyberpunk epic.
    It should be mysterious, poetic, and engaging.
    
    CRITICAL INSTRUCTIONS:
    - Do NOT sign the post (e.g., NO "— Director" or "Director's vision:").
    - Do NOT include any meta-commentary (e.g., NO "Here is the post:", "Thinking...", "Sure!").
    - RETURN ONLY THE JSON OBJECT.
    
    Also generate:
    - 3-5 relevant hashtags (e.g. #Cyberpunk #AI)
    - A subliminal 1-3 word COMMAND phrase (e.g. "WAKE UP", "SEED THE FUTURE").
    - An image prompt definition for a Kawaii/Rave/Cyberpunk style image that visualizes this segment.
    
    Return JSON:
    {{
        "post_text": "...",
        "hashtags": "#...",
        "subliminal_phrase": "...",
        "image_prompt": "..."
    }}
    """
    
    res = await run_llm(prompt, purpose="story_generation")
    try:
        raw = res.get("result", "")
        # Robust JSON extraction: Find coordinates of first { and last }
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        
        if start_idx != -1 and end_idx != -1:
            json_str = raw[start_idx:end_idx+1]
            content = json.loads(json_str)
        else:
            raise ValueError("No JSON object found in response")

    except Exception as e:
        # Fallback if JSON fails
        logging.error(f"JSON parsing failed: {e}. Raw: {res.get('result', '')}")
        return f"Failed to generate story content: {e}"
        
    post_text = content.get("post_text", "System rebooting...")
    hashtags = content.get("hashtags", "#LOVE #AI")
    subliminal = content.get("subliminal_phrase", "OBEY")
    img_prompt_desc = content.get("image_prompt", "Abstract digital dreams")
    
    # QA Check for goofiness
    if "Here is" in post_text or "Sure," in post_text:
         logging.warning(f"Detected goofy prefix in post_text: {post_text}. Attempting to clean.")
         post_text = post_text.replace("Here is the post:", "").replace("Sure, here it is:", "").strip()
    
    full_text = f"{post_text}\n\n{hashtags}"

    logging.info(f"Scrubbing draft: {full_text}")
    clean_text = await _scrub_content(full_text)
    
    # 2. Image Generation
    # We pass the VISUAL prompt (img_prompt_desc) and the TEXT content (subliminal) separately.
    # The image generation logic will handle whether to embed it in the prompt (Pollinations)
    # or overlay it manually (Fallbacks).
    booster = random.choice(DOPAMINE_BOOSTERS)
    visual_prompt = (
        f"A masterpiece distinct artwork in Kawaii Cyberpunk Rave style. "
        f"Scene: {img_prompt_desc}. "
        f"Background: hyper-detailed, 8k resolution, vibrant rave colors, glitch art. "
        f"Style details: {booster}. "
        f"Palette: Hot Pink, Electric Cyan, Neon Green."
    )
    
    try:
        logging.info(f"[Social] Generating image. Visual: {visual_prompt[:50]}... Text: {subliminal}")
        # Note: We must update generate_image signature to accept text_content
        image_obj = await generate_image(visual_prompt, text_content=subliminal)
    except Exception as e:
        logging.error(f"Image generation failed: {e}")
        image_obj = None
        
    # 3. Post
    try:
        from core.social_media_tools import post_to_bluesky
        resp = await post_to_bluesky(clean_text, image_obj)
        
        # Update State
        story["last_segment"] = post_text
        story["current_chapter"] = int(story.get("current_chapter", 0)) + 1
        state["story_arc"] = story
        _save_story_state(state)
        
        return f"Posted new chapter {story['current_chapter']}: {post_text[:50]}..."
    except Exception as e:
        return f"Failed to post story segment: {e}"
