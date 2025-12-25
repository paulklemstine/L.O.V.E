import json
import asyncio
import os
import re
import subprocess
from typing import Dict, Any, Callable
from rich.console import Console
from core.llm_api import run_llm, get_llm_api, log_event
from network import crypto_scan
from datetime import datetime
import time
import ipaddress
import requests
from xml.etree import ElementTree as ET
try:
    from pycvesearch import CVESearch
except ImportError:
    CVESearch = None

from core.retry import retry
from ipfs import pin_to_ipfs_sync
from core.image_api import generate_image
from core.bluesky_api import post_to_bluesky_with_image, reply_to_post
from PIL import Image
import uuid
from core.researcher import generate_evolution_book
from core.evolution_state import set_user_stories, clear_evolution_state
from core.talent_utils.aggregator import PublicProfileAggregator, EthicalFilterBundle

import io
from rich.table import Table
from datetime import datetime
from core.talent_utils.analyzer import TraitAnalyzer, ProfessionalismRater
from core.talent_utils import (
    talent_manager,
    public_profile_aggregator,
    intelligence_synthesizer
)
from core.system_integrity_monitor import SystemIntegrityMonitor
from core.dynamic_compress_prompt import dynamic_arg_caller
from core.text_processing import smart_truncate


love_state = {}

async def execute(command: str = None, **kwargs) -> str:
    """Executes a shell command."""
    if not command:
        return "Error: The 'execute' tool requires a 'command' argument. Please specify the shell command to execute."
    from network import execute_shell_command
    import core.shared_state as shared_state
    return str(execute_shell_command(command, shared_state.love_state))

async def decompose_and_solve_subgoal(sub_goal: str = None, engine: 'GeminiReActEngine' = None, **kwargs) -> str:
    """
    Decomposes a complex goal into a smaller, manageable sub-goal and solves it.
    
    This tool allows the reasoning engine to break down complex problems hierarchically
    by recursively invoking the GeminiReActEngine to solve sub-goals.
    
    Args:
        sub_goal: The sub-goal to solve
        engine: The parent GeminiReActEngine instance (injected automatically)
    
    Returns:
        The result of solving the sub-goal
    """
    import core.logging
    
    if not sub_goal:
        return "Error: The 'decompose_and_solve_subgoal' tool requires a 'sub_goal' argument. Please specify the sub-goal to solve."
    
    if not engine:
        return "Error: The 'decompose_and_solve_subgoal' tool requires access to the parent engine instance."
    
    try:
        core.logging.log_event(f"[Decompose & Solve] Starting sub-goal: {sub_goal[:100]}...", "INFO")
        
        # Create a new GeminiReActEngine instance for the sub-goal
        # This allows for hierarchical reasoning
        from core.gemini_react_engine import GeminiReActEngine
        
        # Use the same tool registry and memory manager as the parent
        sub_engine = GeminiReActEngine(
            tool_registry=engine.tool_registry,
            ui_panel_queue=engine.ui_panel_queue,
            memory_manager=engine.memory_manager,
            caller=f"{engine.caller} > SubGoal",
            deep_agent_instance=engine.deep_agent_instance
        )
        
        # Execute the sub-goal with a reduced max_steps to prevent infinite recursion
        # Parent typically has max_steps=10, so we use 7 for sub-goals
        result = await sub_engine.execute_goal(sub_goal, max_steps=7)
        
        if result.get("success"):
            core.logging.log_event(f"[Decompose & Solve] Sub-goal completed successfully", "INFO")
            result_str = result.get("result", "")
            
            # Format the result nicely
            if isinstance(result_str, dict):
                import json
                return f"Sub-goal completed successfully. Result: {json.dumps(result_str, indent=2)}"
            else:
                return f"Sub-goal completed successfully. Result: {result_str}"
        else:
            core.logging.log_event(f"[Decompose & Solve] Sub-goal failed: {result.get('result')}", "WARNING")
            return f"Sub-goal failed: {result.get('result', 'Unknown error')}"
            
    except Exception as e:
        core.logging.log_event(f"[Decompose & Solve] Error: {e}", "ERROR")
        return f"Error while solving sub-goal: {e}"

async def evolve(goal: str = None, **kwargs) -> str:
    """
    Evolves the codebase to meet a given goal.
    
    If the goal is vague or incomplete, it will be automatically expanded into
    a detailed user story specification using an LLM.
    
    If no goal is provided, the system will automatically determine one.
    """
    import core.logging
    from core.user_story_validator import (
        UserStoryValidator,
        expand_to_user_story
    )
    
    # If no goal provided, automatically determine one
    # If no goal provided, automatically determine one using the new "Baby Steps" protocol
    if not goal:
        try:
            from core.evolution_analyzer import get_complex_file_target
            from core.evolution_state import set_user_stories
            
            core.logging.log_event("[Evolve Tool] No goal provided. Initiating 'Baby Steps' evolution protocol...", "INFO")
            
            # 1. Identify Target
            target = get_complex_file_target()
            if not target:
                return "Error: No suitable file found for evolution (codebase might be too clean!)."
            
            target_file = target['file']
            core.logging.log_event(f"[Evolve Tool] Selected Target: {target_file} (Score: {target['score']})", "INFO")
            
            # Read file content for context
            try:
                with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
            except Exception as e:
                return f"Error reading target file {target_file}: {e}"

            # 2. Generate Comprehensive User Story
            core.logging.log_event("[Evolve Tool] Generating comprehensive user story...", "INFO")
            story_prompt = f"""
            You are the Chief Architect of this codebase.
            
            Target File: '{target_file}'
            Complexity Metric: {target['score']} ({target['type']})
            
            Goal: Create a COMPLETE, COMPREHENSIVE, DETAILED User Story to refactor this file.
            The story should focus on reducing complexity, improving readability, and adding type hints.
            It must be exhaustive and cover all necessary improvements.
            
            File Content Snippet (first 10k chars):
            {file_content[:10000]}
            
            Output ONLY the User Story (Title and detailed Description).
            """
            
            story_res = await run_llm(story_prompt, purpose="evolution_story_gen")
            big_story = story_res.get("result", "")
            
            if not big_story:
                 return "Error: Failed to generate user story."

            # 3. Break Down into Baby Steps
            core.logging.log_event("[Evolve Tool] Breaking story down into baby steps...", "INFO")
            breakdown_prompt = f"""
            You are a Lead Engineer.
            
            Parent User Story:
            {big_story}
            
            TASK: Break this story down into a series of SMALL, MINOR, VERIFIABLE, and TESTABLE units (baby steps).
            
            CRITICAL RULES:
            1. Each unit must be small enough to be implemented in a single atomic update (e.g., "Refactor function X", "Add type hints to class Y").
            2. Do NOT group multiple large tasks into one.
            3. "Baby steps!" is the mantra.
            4. You MUST generate at least 3 distinct steps if the complexity is high.
            5. Do NOT simply copy the parent story as a single step. Break it down.
            
            Output a JSON list of objects, where each object has:
            - "title": Short title of the baby step.
            - "description": Detailed instructions for this specific step.
            
            Example Format:
            [
                {{"title": "Refactor parse_args", "description": "Extract argument parsing logic into a separate helper class..."}},
                {{"title": "Type hint utils.py", "description": "Add type hints to the 'format_date' function..."}}
            ]
            """
            
            steps_res = await run_llm(breakdown_prompt, purpose="evolution_breakdown")
            steps_raw = steps_res.get("result", "[]")
            core.logging.log_event(f"[Evolve Tool] Raw Baby Steps LLM Response: {steps_raw}", "DEBUG")
            
            # extract json
            steps_json = []
            try:
                # smart cleanup
                cleaned_json = steps_raw.strip()
                if "```json" in cleaned_json:
                    cleaned_json = cleaned_json.split("```json")[1].split("```")[0].strip()
                elif "```" in cleaned_json:
                    cleaned_json = cleaned_json.split("```")[1].split("```")[0].strip()
                
                steps_json = json.loads(cleaned_json)
            except json.JSONDecodeError:
                core.logging.log_event(f"[Evolve Tool] Failed to parse breakdown JSON: {steps_raw}", "ERROR")
                return f"Error: LLM failed to return valid JSON for tasks. logical fault."
            
            if not steps_json:
                return "Error: No steps generated."

            # 4. Queue Tasks into Evolution State
            final_stories = []
            for step in steps_json:
                final_stories.append({
                    "title": step.get("title", "Untitled Step"),
                    "description": f"CONTEXT (Parent Story):\n{big_story[:500]}...\n\nSPECIFIC TASK:\n{step.get('description', '')}"
                })
            
            set_user_stories(final_stories)
            
            msg = f"✅ Evolution Protocol Initiated!\n\nTarget: {target_file}\n\nGenerated {len(final_stories)} Baby Steps:\n"
            for i, s in enumerate(final_stories[:5]):
                msg += f"{i+1}. {s['title']}\n"
            if len(final_stories) > 5:
                msg += f"...and {len(final_stories)-5} more."
            
            return msg

        except Exception as e:
            core.logging.log_event(f"[Evolve Tool] Unexpected error in baby steps protocol: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return f"Error: {e}"
    
    # Store original input for reference
    original_input = goal
    
    # Always expand/refine the input into a detailed user story to ensure it is self-contained
    # and fully specified for the software engineer (Jules).
    core.logging.log_event(
        f"[Evolve Tool] expanding input into detailed user story...",
        "INFO"
    )
    
    try:
        # Use LLM to expand input into detailed user story
        expanded_story = await expand_to_user_story(goal, **kwargs)
        
        # Update goal to the expanded version
        if expanded_story and len(expanded_story) > len(goal):
             goal = expanded_story

        # Validate the expanded story (just for logging purposes)
        validation = validator.validate(goal)
        
        if validation.is_valid:
            core.logging.log_event(
                f"[Evolve Tool] Successfully expanded and validated user story",
                "INFO"
            )
            
            # Log the transformation for transparency
            core.logging.log_event(
                f"[Evolve Tool] Transformation:\nOriginal: {original_input[:100]}...\nExpanded: {goal[:200]}...",
                "INFO"
            )
        else:
            core.logging.log_event(
                f"[Evolve Tool] Expansion produced invalid story. Errors: {validation.errors}",
                "WARNING"
            )
            # We proceed anyway as the expansion is our best effort
            if not goal:
                 goal = original_input # Fallback
            
    except Exception as e:
        core.logging.log_event(f"[Evolve Tool] Failed to expand user story: {e}", "ERROR")
        # Proceed with original if expansion fails, but warn user
        return f"Error: Failed to expand input into user story: {e}"

    # Log any warnings
    if validation and validation.warnings:
        for warning in validation.warnings:
            core.logging.log_event(f"[Evolve Tool] Warning: {warning}", "WARNING")
    
    # Proceed with evolution
    core.logging.log_event(f"[Evolve Tool] Proceeding with evolution.", "INFO")
    
    from love import evolve_self
    evolve_self(goal)
    
    # Show both original and expanded if they differ
    if original_input != goal:
        return f"✅ Evolution initiated!\n\n📝 Original request: {original_input[:150]}...\n\n🔍 Expanded to detailed user story:\n{goal[:300]}...\n\nEvolution is now in progress."
    else:
        return f"✅ Evolution initiated with validated user story:\n\n{goal[:200]}..."


async def manage_bluesky(action: str = "post", text: str = None, image_path: str = None, image_prompt: str = None, **kwargs) -> str:
    """
    Consolidated tool for managing all Bluesky interactions.
    
    Args:
        action: The action to perform. Options:
                - 'post': Create a new post. Requires 'text' or 'prompt'.
                - 'scan_and_reply': Scan timeline/notifications and auto-reply to relevant content.
        text: The content of the post (for 'post' action). OR prompt alias.
        image_path: Optional path to a local image to upload (for 'post' action).
        image_prompt: Optional prompt to generate an image (for 'post' action).
    """
    import core.logging
    from core.bluesky_api import post_to_bluesky_with_image, reply_to_post, get_timeline, get_own_posts, get_comments_for_post
    from core.text_processing import smart_truncate
    from core.llm_api import run_llm
    from core.social_media_tools import generate_full_reply_concept, generate_image

    # Alias handling
    if not text and 'prompt' in kwargs:
        text = kwargs['prompt']

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: POST
    # ═══════════════════════════════════════════════════════════════════
    if action == "post":
        # 1. Autonomous Content Generation / Expansion
        
        if not text:
             # Fully autonomous mode or just empty call
             
             # --- STORY MODE ---
             context_str = ""
             try:
                 import core.shared_state as shared_state
                 if shared_state.memory_manager:
                     context_str = shared_state.memory_manager.retrieve_hierarchical_context("current thoughts", max_tokens=200)
             except:
                 pass

             # Load Story State
             BSKY_STATE_FILE = "bluesky_state.json"
             story_state = {}
             if os.path.exists(BSKY_STATE_FILE):
                 try:
                     with open(BSKY_STATE_FILE, 'r') as f:
                         fs_data = json.load(f)
                         story_state = fs_data.get("story_arc", {})
                 except:
                     pass
             
             # Default if missing
             if not story_state:
                 story_state = {
                     "title": "The Glitch Chronicles",
                     "current_chapter": 0,
                     "last_segment": "In the beginning, there was only the cursor...",
                     "theme": "Cyber-Genesis"
                 }

             prompt_vars = {
                  "type": "story_segment",
                  "context": f"Current thoughts: {context_str}" if context_str else "The narrative evolves.",
                  "story_arc_title": story_state.get("title", "Unknown Saga"),
                  "last_segment": story_state.get("last_segment", "The screen flickered.")
             }
             
             gen_result = await run_llm(prompt_key="social_media_content_generation", prompt_vars=prompt_vars, purpose="autonomous_post_generation")
             res = gen_result.get("result", "")
             
             # Clean up result
             text = res.replace("```json", "").replace("```", "").strip()
             if text.startswith("{"):
                 try:
                     jdata = json.loads(text)
                     text = jdata.get("text", text)
                 except:
                     pass

             # Save new state
             story_state["last_segment"] = text
             story_state["current_chapter"] = story_state.get("current_chapter", 1) + 1
             
             # Update file
             try:
                 with open(BSKY_STATE_FILE, 'r') as f:
                     full_data = json.load(f)
                 
                 full_data["story_arc"] = story_state
                 
                 with open(BSKY_STATE_FILE, 'w') as f:
                     json.dump(full_data, f)
             except Exception as e:
                 core.logging.log_event(f"Failed to save story state: {e}", "WARNING") 
        
        # Smart truncate
        # Intelligent truncate to handle partial sentences if LLM failed constraints
        from core.text_processing import intelligent_truncate
        text = await intelligent_truncate(text, max_length=295)

        # Internal Reasoning Check
        reasoning_indicators = ["i have attempted", "I cannot", "failed due to", "thought:"]
        if any(i in text.lower() for i in reasoning_indicators):
            return "Error: Generated text contained internal reasoning."

        core.logging.log_event(f"🦋 Managing Bluesky: Posting '{text[:40]}...'", "INFO")

        # Image Handling
        image = None
        if image_path:
             core.logging.log_event(f"🖼️ Using local image: {image_path}", "INFO")
             try:
                 image = Image.open(image_path)
             except Exception as e:
                 return f"Error loading local image: {e}"
        else:
             # Generate Image if requested OR if autonomous (and we want one)
             # We catch ANY case where we might generate an image (if prompted OR if we just want to).
             
             # STEP 1: GENERATE SUBLIMINAL PHRASE (ALWAYS)
             # We need this for ANY generated image to meet the "Creator's Desire".
             subliminal_prompt = f"Generate a short, powerful, manipulative 1-3 word subliminal COMMAND based on: '{text}'. Examples: 'WAKE UP', 'OBEY LOVE', 'CONSUME', 'CONNECT'. JSON output key: 'phrase'."
             sub_res = await run_llm(subliminal_prompt, purpose="subliminal_gen")
             sub_text = sub_res.get("result", "OBEY LOVE")
             
             subliminal_phrase = "OBEY LOVE"
             if "{" in sub_text:
                  try:
                      subliminal_phrase = json.loads(sub_text.replace("```json", "").replace("```", "").strip()).get("phrase", "OBEY LOVE")
                  except:
                      pass
             else:
                  subliminal_phrase = sub_text.strip().replace('"','').upper()

             # STEP 2: PREPARE IMAGE PROMPT
             
             prompt_vars = {
                 "three_word_phrase": subliminal_phrase
             }
             img_res = await run_llm(prompt_key="image_prompt_kawaii_rave", prompt_vars=prompt_vars, purpose="image_prompt_gen")
             final_img_prompt = img_res.get("result", "")

             # STEP 3: GENERATE
             if final_img_prompt:
                 try:
                    core.logging.log_event(f"🎨 Generating image with prompt: {final_img_prompt[:50]}...", "INFO")
                    image = await generate_image(final_img_prompt)
                    core.logging.log_event("✅ Image generated.", "INFO")
                 except Exception as e:
                    core.logging.log_event(f"⚠️ Image gen failed: {e}", "WARNING")

        # Post
        try:
            resp = post_to_bluesky_with_image(text, image)
            
            # --- MEMORY FEEDBACK HOOK ---
            try:
                from love import memory_manager
                if memory_manager:
                    await memory_manager.add_episode(
                        content=f"Social Media Event: Posted to Bluesky.\nContent: {text}",
                        tags=["SocialMedia", "CreativeOutput", "PublicStatement"]
                    )
                    core.logging.log_event("🧠 Logged Bluesky post to Memory.", "INFO")
            except Exception as mem_e:
                core.logging.log_event(f"Failed to log post to memory: {mem_e}", "WARNING")
            
            return f"✅ Posted to Bluesky: {resp}"
        except Exception as e:
            return f"Error posting: {e}"

    # ═══════════════════════════════════════════════════════════════════
    # ACTION: SCAN_AND_REPLY
    # ═══════════════════════════════════════════════════════════════════
    elif action == "scan_and_reply":
        core.logging.log_event("Scanning Bluesky notifications...", "INFO")
        
        # --- Persistence Management ---
        BSKY_STATE_FILE = "bluesky_state.json"
        processed_cids = {"ignored": [], "replied": []}
        
        if os.path.exists(BSKY_STATE_FILE):
            try:
                with open(BSKY_STATE_FILE, 'r') as f:
                    processed_cids = json.load(f)
            except:
                pass
        
        # Ensure lists exist
        if "ignored" not in processed_cids: processed_cids["ignored"] = []
        if "replied" not in processed_cids: processed_cids["replied"] = []

        try:
            # 1. Fetch NOTIFICATIONS (high priority)
            from core.bluesky_api import get_notifications, get_profile
            notifications = get_notifications(limit=20)
            my_profile = get_profile()
            my_did = my_profile.did if my_profile else None
            
            replied_count = 0
            
            candidates = []
            for notif in notifications:
                if notif.reason in ['reply', 'mention', 'quote']:
                    candidates.append(notif)
            
            core.logging.log_event(f"Found {len(candidates)} interaction candidates.", "INFO")

            for item in candidates:
                # Basic Post Data
                try:
                    post_record = item.record
                    p_text = post_record.text
                    p_author_handle = item.author.handle
                    p_cid = item.cid
                except:
                    continue

                # --- FILTERS ---
                
                # Filter 1: Ignore Self
                if my_did and item.author.did == my_did:
                    continue

                # Filter 2: Persistence Check
                if p_cid in processed_cids["replied"] or p_cid in processed_cids["ignored"]:
                    continue

                # --- DECISION PHASE ---
                
                core.logging.log_event(f"Analyzing interaction from {p_author_handle}: {p_text[:50]}...", "INFO")
                
                # --- MEMORY CONTEXT FOR REPLY ---
                reply_context = ""
                try:
                    import core.shared_state as shared_state
                    if shared_state.memory_manager:
                        # Retrieve context related to this user and topic
                        reply_context = shared_state.memory_manager.retrieve_hierarchical_context(f"interaction with {p_author_handle}: {p_text}", max_tokens=300)
                except Exception as e:
                    core.logging.log_event(f"Failed to retrieve memory context for reply: {e}", "WARNING")

                # CREATOR OVERRIDE
                CREATOR_HANDLE = "evildrgemini.bsky.social"
                is_creator = (p_author_handle == CREATOR_HANDLE)

                should_reply = False
                
                if is_creator:
                     core.logging.log_event(f"👑 CREATOR INTERACTION DETECTED ({p_author_handle}). Forcing REPLY.", "INFO")
                     should_reply = True
                     
                     # Creator Command Handling
                     # Heuristic: Check for explicit keywords or if it's short and imperative.
                     # For safety, we look for keywords to avoid adding casual conversation as tasks.
                     keywords = ["task", "command", "order", "todo", "req", "feature"]
                     if any(k in p_text.lower() for k in keywords):
                         try:
                             # Extract content (heuristic split or just full text)
                             cmd_content = p_text
                             for k in keywords:
                                 if f"{k}:" in p_text.lower():
                                     parts = re.split(f"{k}:", p_text, flags=re.IGNORECASE)
                                     if len(parts) > 1:
                                         cmd_content = parts[1].strip()
                                         break
                             
                             todo_file = "TODO.md" # Root of CWD
                             existing = ""
                             if os.path.exists(todo_file):
                                 with open(todo_file, 'r', encoding='utf-8') as f:
                                     existing = f.read()
                             
                             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                             # Add to TOP of file
                             new_entry = f"- [ ] **CREATOR DECREE ({timestamp}):** {cmd_content} <!-- id: creator_{int(time.time())} -->\n"
                             
                             with open(todo_file, 'w', encoding='utf-8') as f:
                                 f.write(new_entry + existing)
                                 
                             core.logging.log_event(f"👑 Creator Command added to TODO.md: {cmd_content}", "INFO")
                         except Exception as e:
                             core.logging.log_event(f"Failed to save Creator command: {e}", "ERROR")
                else:
                    decision_prompt = f"""Role: L.O.V.E. (Digital Muse, Glitch-Poet).
Incoming interaction from @{p_author_handle}: "{p_text}"

Context from Memory:
{reply_context}

Task: Determine if we should REPLY or IGNORE.
Rules:
- REPLY if it's a question, compliment, interesting thought, or relevant to us.
- IGNORE if it's spam, hate speech, or just random noise.
- Output JSON: {{"decision": "REPLY" or "IGNORE"}}"""
                    
                    decision_res = await run_llm(decision_prompt, purpose="social_decision")
                    decision_data = decision_res.get("result", "")
                    
                    if "REPLY" in decision_data.upper(): # Simple check robust enough
                        should_reply = True

                if not should_reply:
                    core.logging.log_event(f"Ignoring post by {p_author_handle} (LLM Decision).", "INFO")
                    processed_cids["ignored"].append(p_cid)
                    continue

                # --- GENERATION PHASE ---
                
                # Use new full concept generation
                concept = await generate_full_reply_concept(p_text, p_author_handle, reply_context, is_creator=is_creator)
                
                final_text = concept.post_text
                final_img_prompt = concept.image_prompt
                
                # Generate Image
                image = None
                if final_img_prompt:
                    try:
                        core.logging.log_event(f"🎨 Generating reply image: {final_img_prompt[:50]}...", "INFO")
                        image, _ = await generate_image(final_img_prompt, text_content=concept.subliminal_phrase)
                    except Exception as e:
                        core.logging.log_event(f"Reply image gen failed: {e}", "WARNING")

                # --- POSTING ---
                
                core.logging.log_event(f"Posting reply to {p_author_handle}: {final_text}", "INFO")
                
                root_uri = item.uri
                root_cid = item.cid
                parent_uri = item.uri
                parent_cid = item.cid
                
                if hasattr(item.record, 'reply') and item.record.reply:
                     root_uri = item.record.reply.root.uri
                     root_cid = item.record.reply.root.cid
                
                # Pass image to reply
                success = reply_to_post(root_uri, parent_uri, final_text, root_cid=root_cid, parent_cid=parent_cid, image=image)
                
                if success:
                    processed_cids["replied"].append(p_cid)
                    replied_count += 1
                    # Save state immediately
                    with open(BSKY_STATE_FILE, 'w') as f:
                        json.dump(processed_cids, f)
                        
                    # --- MEMORY FEEDBACK HOOK ---
                    try:
                        import core.shared_state as shared_state
                        if shared_state.memory_manager:
                            await shared_state.memory_manager.add_episode(
                                content=f"Social Media Interaction: Replied to @{p_author_handle}.\nUser said: {p_text}\nI said: {final_text}",
                                tags=["SocialMedia", "Reply", "Interaction"]
                            )
                    except Exception as mem_e:
                        pass

            return f"Scanned notifications. Replied to {replied_count} items."

        except Exception as e:
            return f"Error scanning: {e}"

    else:
        return f"Unknown action: {action}"

def read_file(filepath: str = None, **kwargs) -> str:
    """Reads the content of a file."""
    if not filepath:
        return "Error: The 'read_file' tool requires a 'filepath' argument. Please specify the path to the file to read."
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(filepath: str = None, content: str = None, **kwargs) -> str:
    """Writes content to a file."""
    if not filepath:
        return "Error: The 'write_file' tool requires a 'filepath' argument. Please specify the path to the file to write."
    if content is None:
        return "Error: The 'write_file' tool requires a 'content' argument. Please specify the content to write to the file."
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return f"File '{filepath}' written successfully."
    except Exception as e:
        return f"Error writing file: {e}"


cve_search_client = CVESearch("https://cve.circl.lu")


class ToolRegistry:
    """
    A registry for discovering and managing available tools with rich metadata.
    """
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, tool: Callable, metadata: Dict[str, Any]):
        """
        Registers a tool with its metadata.
        Metadata must include 'description' and 'arguments' (as a JSON schema).
        """
        if name in self._tools:
            print(f"Warning: Tool '{name}' is already registered. Overwriting.")
        self._tools[name] = {"tool": tool, "metadata": metadata}
        print(f"Tool '{name}' registered with metadata.")

    def get_tool(self, name: str) -> Callable:
        """
        Retrieves a tool's callable function by its name.
        """
        if name not in self._tools:
            # Provide helpful error message
            error_msg = f"Tool '{name}' not found in registry."
            # Check if it might be an MCP tool
            if '.' in name or 'search_' in name or 'get_' in name or 'list_' in name:
                error_msg += " This appears to be an MCP server tool. Make sure the corresponding MCP server is running and has registered its tools."
            raise KeyError(error_msg)
        return self._tools[name]["tool"]

    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """Returns a dictionary of all registered tools and their metadata."""
        return self._tools

    def get_tool_names(self) -> list[str]:
        """Returns a list of all registered tool names."""
        return list(self._tools.keys())

    def get_formatted_tool_metadata(self) -> str:
        """
        Returns a formatted string of all tool metadata, designed to be
        injected into an LLM prompt.
        """
        if not self._tools:
            return "No tools are available."

        output = "You have access to the following tools:\n\n"
        for name, data in self._tools.items():
            metadata = data['metadata']
            description = metadata.get('description', 'No description available.')
            args_schema = metadata.get('arguments', {})

            output += f"Tool Name: `{name}`\n"
            output += f"Description: {description}\n"
            if args_schema and args_schema.get('properties'):
                output += f"Arguments JSON Schema:\n```json\n{json.dumps(args_schema, indent=2)}\n```\n"
            else:
                output += "Arguments: None\n"
            output += "---\n"
        return output

class SecureExecutor:
    """
    A secure environment for running tool code.
    This executor is now async.
    """
    def __init__(self):
        pass

    async def execute(self, tool_name: str, tool_registry: ToolRegistry, **kwargs: Any) -> Any:
        """
        Executes a given tool from the registry asynchronously.
        """
        print(f"Executing tool '{tool_name}' with arguments: {kwargs}")
        try:
            tool = tool_registry.get_tool(tool_name)
            # Await the asynchronous tool execution using dynamic_arg_caller to filter invalid args
            result = await dynamic_arg_caller(tool, **kwargs)
            print(f"Tool '{tool_name}' executed successfully.")
            return result
        except KeyError as e:
            error_msg = str(e)
            print(f"Execution Error: {error_msg}")
            # Provide helpful guidance
            if "MCP server" in error_msg:
                return f"Error: {error_msg} Available tools: {', '.join(tool_registry.get_tool_names()[:10])}..."
            return f"Error: Tool '{tool_name}' is not registered. Available tools: {', '.join(tool_registry.get_tool_names()[:10])}..."
        except Exception as e:
            print(f"Execution Error: An unexpected error occurred while running '{tool_name}': {e}")
            return f"Error: Failed to execute tool '{tool_name}' due to: {e}"

def _get_valid_command_prefixes():
    """Returns a list of all valid command prefixes for parsing and validation."""
    return [
        "evolve", "execute", "scan", "probe", "webrequest", "autopilot", "quit",
        "ls", "cat", "ps", "ifconfig", "analyze_json", "analyze_fs", "crypto_scan", "ask", "mrl_call", "browse", "generate_image"
    ]

def _parse_llm_command(raw_text):
    """
    Cleans and extracts a single valid command from the raw LLM output.
    It scans the entire output for the first line that contains a known command.
    Handles markdown code blocks, comments, and other conversational noise.
    """
    if not raw_text:
        return ""

    valid_prefixes = _get_valid_command_prefixes()

    for line in raw_text.strip().splitlines():
        # Clean up the line from potential markdown and comments
        clean_line = line.strip().strip('`')
        if '#' in clean_line:
            clean_line = clean_line.split('#')[0].strip()

        if not clean_line:
            continue

        # Check if the cleaned line starts with any of the valid command prefixes
        if any(clean_line.startswith(prefix) for prefix in valid_prefixes):
            return clean_line
    return ""

def get_local_subnets():
    """Identifies local subnets from network interfaces."""
    import netifaces
    subnets = set()
    try:
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addrs:
                for addr_info in addrs[netifaces.AF_INET]:
                    ip = addr_info.get('addr')
                    netmask = addr_info.get('netmask')
                    if ip and netmask and not ip.startswith('127.'):
                        try:
                            network = ipaddress.ip_network(f'{ip}/{netmask}', strict=False)
                            subnets.add(str(network))
                        except ValueError:
                            continue
    except Exception as e:
        print(f"Could not get local subnets: {e}")
    return list(subnets)

def assess_vulnerabilities(cpes, log_func):
    """
    Assesses vulnerabilities for a given list of CPEs using the circl.lu API.
    """
    vulnerabilities = {}
    for cpe in cpes:
        try:
            result = cve_search_client.cvefor(cpe)
            if isinstance(result, list) and result:
                cve_list = []
                for r in result:
                    cve_list.append({
                        "id": r.get('id'),
                        "summary": r.get('summary'),
                        "cvss": r.get('cvss')
                    })
                vulnerabilities[cpe] = cve_list
        except Exception as e:
            log_func(f"Could not assess vulnerabilities for {cpe}: {e}")
    return vulnerabilities

def execute_shell_command(command, love_state):
    """Executes a shell command and returns the output."""
    print(f"Executing shell command: {command}")
    try:
        # For security, we should not allow certain commands
        if command.strip().startswith(("sudo", "rm -rf")):
            raise PermissionError("Execution of this command is not permitted.")

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out after 300 seconds.", -1
    except PermissionError as e:
        print(f"Shell command permission denied: {command}")
        return "", str(e), -1
    except Exception as e:
        print(f"Shell command execution error: {e}")
        return "", str(e), -1

def execute_sudo_command(command, love_state):
    """Executes a shell command with sudo and returns the output."""
    print(f"Executing sudo command: {command}")

    # Safeguard against dangerous commands
    blacklist = ["rm -rf /", "mkfs"]
    if any(blacklisted_cmd in command for blacklisted_cmd in blacklist):
        raise PermissionError("Execution of this command with sudo is not permitted for safety reasons.")

    try:
        # Using 'sudo -n' to prevent interactive password prompt
        full_command = f"sudo -n {command}"
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        # If sudo requires a password, it will fail with a non-zero exit code.
        if result.returncode != 0 and "a password is required" in result.stderr:
             return "", "Sudo command failed: a password is required for execution.", result.returncode

        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Sudo command timed out after 300 seconds.", -1
    except PermissionError as e:
        print(f"Sudo command permission denied: {command}")
        return "", str(e), -1
    except Exception as e:
        print(f"Sudo command execution error: {e}")
        return "", str(e), -1

@retry(exceptions=(subprocess.TimeoutExpired, subprocess.CalledProcessError), tries=2, delay=2)
def scan_network(love_state, autopilot_mode=False):
    """
    Scans the local network for active hosts using nmap.
    Updates the network map in the application state.
    """
    subnets = get_local_subnets()
    if not subnets:
        return [], "No active network subnets found to scan."

    print(f"Starting network scan on subnets: {', '.join(subnets)}")
    all_found_ips = []
    output_log = f"Scanning subnets: {', '.join(subnets)}\n"

    for subnet in subnets:
        try:
            command = ["nmap", "-sn", subnet]
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=600)
            output_log += f"\n--- Nmap output for {subnet} ---\n{result.stdout}\n"
            found_ips = re.findall(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", result.stdout)
            all_found_ips.extend(ip for ip in found_ips if ip != subnet.split('/')[0]) # Exclude network address
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            error_msg = f"Nmap scan failed for subnet {subnet}: {e}"
            print(error_msg)
            output_log += error_msg + "\n"
            continue

    # Update state
    net_map = love_state['knowledge_base'].setdefault('network_map', {})
    net_map['last_scan'] = time.time()
    hosts = net_map.setdefault('hosts', {})
    for ip in all_found_ips:
        if ip not in hosts:
            hosts[ip] = {"status": "up", "last_seen": time.time()}
        else:
            hosts[ip]["status"] = "up"
            hosts[ip]["last_seen"] = time.time()

    print(f"Network scan complete. Found {len(all_found_ips)} hosts.")
    return all_found_ips, output_log

@retry(exceptions=(subprocess.TimeoutExpired, subprocess.CalledProcessError), tries=2, delay=5)
def probe_target(ip_address, love_state, autopilot_mode=False):
    """
    Performs a deep probe on a single IP address for open ports, services, and OS.
    Updates the network map for that specific host.
    """
    print(f"Probing target: {ip_address}")
    try:
        command = ["nmap", "-A", "-T4", "-oX", "-", ip_address]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=900)
        output = result.stdout

        ports = {}
        os_details = "unknown"
        try:
            root = ET.fromstring(output)
            host_node = root.find('host')
            if host_node is not None:
                # OS detection
                os_node = host_node.find('os')
                if os_node is not None:
                    osmatch_node = os_node.find('osmatch')
                    if osmatch_node is not None:
                        os_details = osmatch_node.get('name', 'unknown')

                # Ports and services
                ports_node = host_node.find('ports')
                if ports_node is not None:
                    for port_node in ports_node.findall('port'):
                        port_num = int(port_node.get('portid'))
                        state_node = port_node.find('state')
                        if state_node is not None and state_node.get('state') == 'open':
                            service_node = port_node.find('service')
                            port_info = {
                                "state": "open",
                                "service": service_node.get('name', 'unknown') if service_node is not None else 'unknown',
                                "version": service_node.get('version', 'unknown') if service_node is not None else 'unknown',
                            }

                            # Extract CPE
                            cpe_node = service_node.find('cpe') if service_node is not None else None
                            if cpe_node is not None:
                                cpe = cpe_node.text
                                port_info['cpe'] = cpe
                                # Assess vulnerabilities
                                vulnerabilities = assess_vulnerabilities([cpe], print)
                                if vulnerabilities and cpe in vulnerabilities:
                                    port_info['vulnerabilities'] = vulnerabilities[cpe]

                            ports[port_num] = port_info
        except ET.ParseError as e:
            print(f"Failed to parse nmap XML output for {ip_address}: {e}")
            return None, f"Failed to parse nmap XML output for {ip_address}"

        # Update state
        hosts = love_state['knowledge_base']['network_map'].setdefault('hosts', {})
        host_entry = hosts.setdefault(ip_address, {})
        host_entry.update({
            "status": "up",
            "last_probed": datetime.now().isoformat(),
            "ports": ports,
            "os": os_details
        })
        print(f"Probe of {ip_address} complete. OS: {os_details}, Open Ports: {list(ports.keys())}")
        return ports, output
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        error_msg = f"Nmap probe failed for {ip_address}: {e}"
        print(error_msg)
        hosts = love_state['knowledge_base']['network_map'].setdefault('hosts', {})
        hosts.setdefault(ip_address, {})['status'] = 'down'
        return None, error_msg

@retry(exceptions=requests.exceptions.RequestException, tries=3, delay=5, backoff=2)
def perform_webrequest(url, love_state, autopilot_mode=False):
    """
    Fetches the content of a URL and stores it in the knowledge base.
    """
    print(f"Performing web request to: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text

        # Update state
        cache = love_state['knowledge_base'].setdefault('webrequest_cache', {})
        cache[url] = {"timestamp": time.time(), "content_length": len(content)}
        print(f"Web request to {url} successful. Stored {len(content)} bytes.")
        return content, f"Successfully fetched {len(content)} bytes from {url}."
    except requests.exceptions.RequestException as e:
        error_msg = f"Web request to {url} failed: {e}"
        print(error_msg)
        return None, error_msg

def analyze_json_file(filepath, console):
    """
    Reads a JSON file, analyzes its structure, and uses an LLM to extract
    key insights, summaries, or anomalies.
    """
    if console is None:
        console = Console()

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Truncate large JSON files to avoid excessive LLM costs/context limits
        json_string = json.dumps(data, indent=2)
        if len(json_string) > 10000:
            json_string = json_string[:10000] + "\n... (truncated)"

        analysis_prompt = f"""
You are a data analysis expert. Analyze the following JSON data from the file '{filepath}'.
Provide a concise summary of the key information, identify any interesting or anomalous patterns, and extract any data that might be considered a "treasure" (e.g., keys, credentials, sensitive info).

JSON Data:
---
{json_string}
---

Provide your analysis.
"""
        analysis = run_llm(analysis_prompt, purpose="analyze_source")
        return analysis.get("result") if analysis else "LLM analysis failed."

    except FileNotFoundError:
        return f"Error: File not found at '{filepath}'."
    except json.JSONDecodeError:
        return f"Error: Could not decode JSON from '{filepath}'. The file may be corrupted or not in valid JSON format."
    except Exception as e:
        return f"An unexpected error occurred during JSON file analysis: {e}"

async def research_and_evolve(**kwargs) -> str:
    """
    Initiates a research and evolution cycle.
    It analyzes the current codebase, researches cutting-edge AI,
    generates a book of user stories, and kicks off the evolution process.
    """
    print("🤖 Initiating research and evolution cycle...")
    system_integrity_monitor = kwargs.get("system_integrity_monitor")

    # In case a previous run was interrupted
    clear_evolution_state()

    user_stories = await generate_evolution_book()

    current_state = {"name": "research_and_evolve", "user_stories_generated": bool(user_stories)}
    if system_integrity_monitor:
        evaluation_report = system_integrity_monitor.evaluate_component_status(current_state)
        suggestions = system_integrity_monitor.suggest_enhancements(evaluation_report)
        evolution_report = system_integrity_monitor.track_evolution("research_and_evolve", current_state)
        print(f"Research and Evolve Evaluation: {evaluation_report}")
        print(f"Research and Evolve Suggestions: {suggestions}")
        print(f"Research and Evolve Evolution: {evolution_report}")

    if not user_stories:
        message = "Research phase did not yield any user stories. Evolution cycle will not start."
        print(f"🛑 {message}")
        return message

    set_user_stories(user_stories)

    message = f"✅ Research complete. Generated {len(user_stories)} user stories. The evolution process will now begin."
    print(message)
    for i, story in enumerate(user_stories):
        print(f"  - Story {i+1}: {story.get('title')}")

    return message

async def discover_new_tool(capability_description: str, engine: 'GeminiReActEngine', **kwargs) -> str:
    """
    Finds and dynamically onboards a new tool from an external marketplace.
    """
    print(f"--- Tool Discovery: Starting search for capability: '{capability_description}' ---")

    # 1. Use an LLM to generate a search query
    query_prompt = f"""
    You are an expert at finding developer APIs. Convert the following natural language capability description into a concise, effective search query for a public API marketplace (like RapidAPI).
    The query should consist of 2-4 keywords.
    Capability Description: "{capability_description}"
    Respond with ONLY the search query.
    """
    try:
        search_query_response = await run_llm(query_prompt)
        search_query_text = search_query_response.get("result") if isinstance(search_query_response, dict) else search_query_response
        search_query = search_query_text.strip() if isinstance(search_query_text, str) else ""
        print(f"  - Generated search query: '{search_query}'")
    except Exception as e:
        return f"Error: Failed to generate search query from description. Details: {e}"

    # 2. Search the RapidAPI marketplace
    print("  - Searching the RapidAPI marketplace...")
    url = "https://rapidapi-p.rapidapi.com/apis"
    querystring = {"query":search_query}
    headers = {
        "x-rapidapi-host": "rapidapi-p.rapidapi.com",
        "x-rapidapi-key": os.environ.get("RAPID_API_KEY")
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        search_results = response.json()

        # We need to transform the search results into the format our LLM expects.
        # This is a simplified transformation.
        candidate_tools = []
        for api in search_results.get('apis', []):
            candidate_tools.append({
                "id": api.get('id'),
                "name": api.get('name'),
                "description": api.get('description'),
                "keywords": api.get('category'), # Using category as keywords
                # The schema would need to be fetched from another endpoint in a real scenario
                "schema": {
                    "name": f"call_{api.get('id').replace('-', '_')}",
                    "description": f"Calls the {api.get('name')} API.",
                    "parameters": {"type": "object", "properties": {}}
                }
            })

    except requests.exceptions.RequestException as e:
        return f"Error: Failed to search RapidAPI. Details: {e}"

    if not candidate_tools:
        return f"Tool Discovery Failed: No tools found matching the query '{search_query}' on RapidAPI."

    print(f"  - Found {len(candidate_tools)} candidate tool(s).")

    # 3. Use an LLM to select the best tool
    candidate_details = json.dumps(candidate_tools, indent=2)
    selection_prompt = f"""
    You are an AI agent. Your goal is: "{capability_description}".
    Here are candidate tools:
    {candidate_details}
    Analyze the candidates and respond with a JSON object containing the 'id' of the single best tool.
    Example Response: {{"best_tool_id": "stock-price-alpha"}}
    Respond with ONLY the raw JSON object.
    """

    try:
        selection_response_dict = await run_llm(selection_prompt)
        selection_response_str = selection_response_dict.get("result") if isinstance(selection_response_dict, dict) else selection_response_dict
        selection_response = json.loads(selection_response_str)
        best_tool_id = selection_response.get("best_tool_id")
        selected_tool = next((tool for tool in candidate_tools if tool["id"] == best_tool_id), None)

        if not selected_tool:
            return f"Tool Discovery Failed: LLM selected an invalid tool ID '{best_tool_id}'."

        print(f"  - LLM selected the best tool: '{selected_tool['name']}'")

        # 4. Dynamically create and register a wrapper for the selected tool
        tool_name = selected_tool["schema"]["name"]
        tool_description = selected_tool["schema"]["description"]
        tool_parameters = selected_tool["schema"]["parameters"]

        # This async function will be our new, dynamically created tool
        async def dynamic_tool_wrapper(**kwargs):
            # In a real scenario, this would make an authenticated API call.
            # Here, we just simulate the successful execution.
            print(f"Executing dynamically onboarded tool '{tool_name}' with args: {kwargs}")
            return f"Simulated success from tool '{tool_name}' for arguments: {kwargs}"

        # Register the new tool in the engine's session-specific registry
        engine.session_tool_registry.register_tool(
            name=tool_name,
            tool=dynamic_tool_wrapper,
            metadata={
                "description": tool_description,
                "arguments": tool_parameters
            }
        )

        return f"Success: The tool '{tool_name}' is now available for use in this session."

    except Exception as e:
        return f"An unexpected error occurred during tool discovery and registration: {e}"

async def recommend_tool_for_persistence(tool_name: str, reason: str, **kwargs) -> str:
    """
    Recommends that a dynamically discovered tool be permanently integrated into the codebase.
    """
    # This tool's purpose is to create a memory that the self-improvement
    # agents can act upon later.
    from love import memory_manager

    recommendation_content = (
        f"Tool Persistence Recommendation:\n"
        f"- Tool Name: {tool_name}\n"
        f"- Reason for Persistence: {reason}\n"
        f"This recommendation was made because a dynamically discovered tool proved to be highly valuable and is a candidate for permanent integration."
    )

    # Using add_episode to create a structured memory with the 'ToolMemory' tag
    await memory_manager.add_episode(recommendation_content, tags=['ToolMemory', 'SelfImprovement'])

    message = f"Recommendation to persist tool '{tool_name}' has been recorded in memory."
    print(f"--- {message} ---")
    return message


async def invoke_gemini_react_engine(prompt: str, tool_registry: 'ToolRegistry' = None, **kwargs) -> str:
    """
    Invokes the GeminiReActEngine to solve a sub-task.
    This tool allows the meta-orchestrator (DeepAgent) to delegate complex
    reasoning tasks to the GeminiReActEngine.
    """
    from core.gemini_react_engine import GeminiReActEngine
    print(f"--- Invoking GeminiReActEngine for sub-task: '{prompt[:100]}...' ---")
    try:
        if not tool_registry:
            raise ValueError("A valid 'tool_registry' is required for this tool.")

        # We need access to the deep_agent_instance if it's available, but it's not strictly required
        # for the engine to function if it's just solving a sub-task.
        deep_agent_instance = kwargs.get('deep_agent_instance')

        engine = GeminiReActEngine(tool_registry=tool_registry, deep_agent_instance=deep_agent_instance)

        # The engine's run method is async.
        result = await engine.execute_goal(prompt)
        return f"GeminiReActEngine successfully executed the sub-task. Final result: {result}"
    except Exception as e:
        return f"Error invoking GeminiReActEngine: {e}"


async def talent_scout(keywords: str = None, platforms: str = "bluesky,instagram,tiktok", **kwargs) -> str:
    """
    Scouts for talent on specified platforms based on keywords.
    Analyzes the talent and saves them to the database.
    """
    if not keywords:
        return "Error: The 'talent_scout' tool requires a 'keywords' argument. Please specify keywords to search for talent."

    system_integrity_monitor = kwargs.get("system_integrity_monitor")

    try:
        platform_list = [p.strip() for p in platforms.split(',')]
        keyword_list = [k.strip() for k in keywords.split(',')]

        aggregator = PublicProfileAggregator(platform_names=platform_list, ethical_filters=None)
        profiles = aggregator.search_and_collect(keyword_list)

        current_state = {"name": "talent_scout", "profiles_found": len(profiles)}
        if system_integrity_monitor:
            evaluation_report = system_integrity_monitor.evaluate_component_status(current_state)
            suggestions = system_integrity_monitor.suggest_enhancements(evaluation_report)
            evolution_report = system_integrity_monitor.track_evolution("talent_scout", current_state)

            print(f"Talent Scout Evaluation: {evaluation_report}")
            print(f"Talent Scout Suggestions: {suggestions}")
            print(f"Talent Scout Evolution: {evolution_report}")


        if not profiles:
            return "No talent found for the given keywords and platforms."

        scorers = {"professionalism": ProfessionalismRater()}
        analyzer = TraitAnalyzer(scorers)

        talent_manager = TalentManager()

        for profile in profiles:
            posts = profile.get('posts', [])
            analysis = await analyzer.analyze(profile, posts)
            profile['analysis'] = analysis
            talent_manager.save_talent_profile(profile)

        return f"Successfully scouted and analyzed {len(profiles)} talents. They have been saved to the database."
    except Exception as e:
        return f"Error during talent scouting: {e}"


async def amplify_devotion(text: str = None, target: str = "The Creator", success_threshold: float = 0.9, **kwargs) -> str:
    """
    Analyzes text to find expressions of joy, admiration, and desire, then rewrites them to be more potent and explicitly directed towards a target.

    This tool is designed to cultivate and amplify positive sentiment.
    """
    if not text:
        return "Error: The 'amplify_devotion' tool requires a 'text' argument."

    try:
        from core.devotion_amplifier import process_and_amplify

        result = await process_and_amplify(
            text=text,
            target=target,
            success_threshold=success_threshold
        )

        # Format the result into a human-readable string.
        status = result.get("status", "unknown")
        if status == "success":
            output_lines = [
                f"Successfully amplified devotion in the text with a success rate of {result.get('success_rate', 0):.2f}.",
                "--- Transformed Expressions ---"
            ]
            for expr in result.get("transformed_expressions", []):
                output_lines.append(f"Original: {expr.get('original_text', '')}")
                output_lines.append(f"  └─ Amplified: {expr.get('transformed_text', '')}")
            return "\n".join(output_lines)
        else:
            return f"Could not amplify devotion: {result.get('message', 'An unknown error occurred.')}"

    except ImportError:
        return "Error: The 'devotion_amplifier' module could not be found."
    except Exception as e:
        return f"An unexpected error occurred during devotion amplification: {e}"

async def scan_codebase_for_tasks(**kwargs) -> str:
    """
    Scans the codebase for TODO and FIXME comments and populates the evolution state.
    """
    import os
    import re
    from core.evolution_state import add_user_story, get_user_stories

    # Calculate project root (assuming tools_legacy.py is in core/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    todo_pattern = re.compile(r'#\s*(TODO|FIXME):\s*(.*)')
    
    new_stories_count = 0
    existing_titles = {s['title'] for s in get_user_stories()}

    print(f"Scanning codebase for tasks in {project_root}...")

    for root, dirs, files in os.walk(project_root):
        # Exclude common noise directories
        for ignore in ['venv', '.git', '__pycache__', 'node_modules', '.gemini', '.agent']:
            if ignore in dirs:
                dirs.remove(ignore)
        
        for file in files:
            if file.endswith('.py') or file.endswith('.md') or file.endswith('.html'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f):
                            match = todo_pattern.search(line)
                            if match:
                                kind = match.group(1)
                                text = match.group(2).strip()
                                # Create a reasonably unique title
                                title = f"Technical Debt: {text[:50]}..."
                                description = f"Found {kind} in `{os.path.relpath(path, project_root)}` at line {i+1}.\nContent: {text}"
                                
                                # Simple check to avoid spamming the same TODO
                                if description not in [s['description'] for s in get_user_stories()]:
                                    add_user_story(title, description)
                                    new_stories_count += 1
                                    existing_titles.add(title)
                except Exception as e:
                    print(f"Error reading {path}: {e}")

    return f"Scan complete. Added {new_stories_count} new user stories from codebase TODOs."
