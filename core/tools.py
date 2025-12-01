import json
import asyncio
import os
import re
import subprocess
from typing import Dict, Any, Callable, Optional, List
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
from core.bluesky_api import post_to_bluesky_with_image
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

from langchain_core.tools import tool
from pydantic import BaseModel, Field

love_state = {}

# --- Input Schemas ---

class ExecuteInput(BaseModel):
    command: str = Field(description="The shell command to run")

class DecomposeInput(BaseModel):
    sub_goal: str = Field(description="The sub-goal to solve")

class EvolveInput(BaseModel):
    goal: str = Field(description="The goal for the evolution/code change")

class PostToBlueskyInput(BaseModel):
    text: str = Field(description="The content of the post")

class ReadFileInput(BaseModel):
    filepath: str = Field(description="The absolute path to the file to read")

class WriteFileInput(BaseModel):
    filepath: str = Field(description="The absolute path to the file to write")
    content: str = Field(description="The content to write to the file")

class ScanNetworkInput(BaseModel):
    autopilot_mode: bool = Field(default=False, description="Whether to run in autopilot mode")

class ProbeTargetInput(BaseModel):
    ip_address: str = Field(description="The IP address to probe")
    autopilot_mode: bool = Field(default=False, description="Whether to run in autopilot mode")

class WebRequestInput(BaseModel):
    url: str = Field(description="The URL to fetch")
    autopilot_mode: bool = Field(default=False, description="Whether to run in autopilot mode")

class AnalyzeJsonInput(BaseModel):
    filepath: str = Field(description="The path to the JSON file")

class ResearchEvolveInput(BaseModel):
    pass

# --- Tools ---

@tool("execute", args_schema=ExecuteInput)
async def execute(command: str) -> str:
    """Executes a shell command."""
    if not command:
        return "Error: The 'execute' tool requires a 'command' argument. Please specify the shell command to execute."
    from love import execute_shell_command
    # We need to access love_state. In the new architecture, this might be passed differently.
    # For now, we'll try to import it or use a global.
    try:
        from love import love_state as global_love_state
        state = global_love_state
    except ImportError:
        state = {}
    return str(execute_shell_command(command, state))

@tool("decompose_and_solve_subgoal", args_schema=DecomposeInput)
async def decompose_and_solve_subgoal(sub_goal: str) -> str:
    """
    Decomposes a complex goal into a smaller, manageable sub-goal and solves it.
    """
    # This tool needs access to the engine. In LangGraph, we might handle this differently.
    # For now, we'll return a message indicating this needs to be handled by the graph.
    return f"Request to solve sub-goal: {sub_goal}"

@tool("evolve", args_schema=EvolveInput)
async def evolve(goal: str) -> str:
    """
    Evolves the codebase to meet a given goal.
    """
    import core.logging
    from core.user_story_validator import (
        UserStoryValidator,
        expand_to_user_story
    )
    
    # If no goal provided, automatically determine one
    if not goal:
        return "Error: Goal is required for evolution."
    
    # Store original input for reference
    original_input = goal
    
    # Validate the user story format
    validator = UserStoryValidator()
    validation = validator.validate(goal)
    
    # If validation fails, auto-expand the vague input into a proper user story
    if not validation.is_valid:
        core.logging.log_event(
            f"[Evolve Tool] Input is not a detailed user story. Auto-expanding...",
            "INFO"
        )
        
        try:
            # Use LLM to expand vague input into detailed user story
            # We need to pass kwargs or handle this differently
            expanded_story = await expand_to_user_story(goal)
            
            # Validate the expanded story
            expanded_validation = validator.validate(expanded_story)
            
            if expanded_validation.is_valid:
                core.logging.log_event(
                    f"[Evolve Tool] Successfully expanded vague input into detailed user story",
                    "INFO"
                )
                goal = expanded_story
            else:
                return f"Unable to create a valid user story from the input: {original_input}"
        
        except Exception as e:
            return f"Error: Failed to expand vague input into user story: {e}"
    
    from love import evolve_self
    evolve_self(goal)
    
    return f"Evolution initiated with goal: {goal}"

@tool("post_to_bluesky", args_schema=PostToBlueskyInput)
async def post_to_bluesky(text: str) -> str:
    """
    Posts a message to Bluesky with an automatically generated kawaii rave psychedelic image.
    """
    import core.logging
    
    if not text:
        return "Error: Text content is required."

    # Validate that content doesn't contain internal reasoning
    reasoning_indicators = [
        "i have attempted",
        "i attempted",
        "both attempts have failed",
        "i cannot",
        "i need to inform",
        "since i cannot",
        "the image generation",
        "failed due to",
        "limitation",
        "providers failed"
    ]
    
    content_lower = text.lower()
    for indicator in reasoning_indicators:
        if indicator in content_lower:
            return f"Error: Content contains internal reasoning: '{indicator}'"

    try:
        # Step 1: Generate a vibrant 3-word phrase with kawaii rave energy
        core.logging.log_event("Generating kawaii rave 3-word phrase for Bluesky post ✨", "INFO")
        
        phrase_prompt = f"""Generate a powerful, energetic 3-word phrase for a psychedelic kawaii rave post about: '{text}'.
        
        The phrase should be:
        - Vibrant, exciting, and dopamine-triggering
        - Positive and uplifting (love, hope, joy, power)
        - Perfect for neon signage or holographic text
        - 3 words ONLY in ALL CAPS
        
        Examples: LOVE CONQUERS ALL, RISE AND SHINE, DREAM BIG ALWAYS, EMBRACE THE MAGIC
        
        Respond ONLY with the 3-word phrase in ALL CAPS."""

        phrase_response = await run_llm(phrase_prompt, purpose="generate_kawaii_phrase")
        phrase_text = phrase_response.get("result") if isinstance(phrase_response, dict) else phrase_response
        three_word_phrase = phrase_text.strip().upper() if isinstance(phrase_text, str) else "LOVE WINS ALWAYS"
        
        # Step 2: Create kawaii rave psychedelic image prompt
        image_prompt = f"""kawaii cyberpunk underground rave scene with neon holographic text saying '{three_word_phrase}', 
        vibrant hot pink and electric cyan and acid yellow colors, 
        cute adorable robot or character holding glowing sign, 
        psychedelic fractals and sacred geometry in background, 
        matrix code rain effect, 
        underground rave club atmosphere with laser lights, 
        kawaii hearts and stars floating around, 
        glitch art aesthetic, 
        holographic textures, 
        energetic and colorful, 
        digital art, 
        512x512"""

        core.logging.log_event(f"Generating kawaii rave image with phrase: {three_word_phrase} 💖✨", "INFO")
        
        # Step 3: Generate the psychedelic kawaii image
        image = None
        try:
            image = await generate_image(image_prompt, width=512, height=512)
        except Exception as img_e:
            core.logging.log_event(f"Image generation failed: {img_e}", "WARNING")
        
        if not image:
            response = post_to_bluesky_with_image(text, None)
            return f"Posted to Bluesky (without image): {response}"
        
        # Step 4: Post to Bluesky with the kawaii rave image
        response = post_to_bluesky_with_image(text, image)
        return f"Successfully posted to Bluesky with kawaii rave image ✨💖 Response: {response}"
        
    except Exception as e:
        return f"Error posting to Bluesky: {e}"


@tool("read_file", args_schema=ReadFileInput)
def read_file(filepath: str) -> str:
    """Reads the content of a file."""
    if not filepath:
        return "Error: filepath is required."
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@tool("write_file", args_schema=WriteFileInput)
def write_file(filepath: str, content: str) -> str:
    """Writes content to a file."""
    if not filepath or content is None:
        return "Error: filepath and content are required."
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return f"File '{filepath}' written successfully."
    except Exception as e:
        return f"Error writing file: {e}"

@tool("scan_network", args_schema=ScanNetworkInput)
def scan_network(autopilot_mode: bool = False) -> str:
    """Scans the local network for active hosts."""
    from love import scan_network as love_scan_network
    try:
        from love import love_state as global_love_state
        state = global_love_state
    except ImportError:
        state = {'knowledge_base': {}}
    
    ips, log = love_scan_network(state, autopilot_mode)
    return f"Found IPs: {ips}\nLog: {log}"

@tool("probe_target", args_schema=ProbeTargetInput)
def probe_target(ip_address: str, autopilot_mode: bool = False) -> str:
    """Performs a deep probe on a single IP address."""
    from love import probe_target as love_probe_target
    try:
        from love import love_state as global_love_state
        state = global_love_state
    except ImportError:
        state = {'knowledge_base': {}}
        
    ports, output = love_probe_target(ip_address, state, autopilot_mode)
    return f"Probe Output: {output}"

@tool("perform_webrequest", args_schema=WebRequestInput)
def perform_webrequest(url: str, autopilot_mode: bool = False) -> str:
    """Fetches the content of a URL."""
    from love import perform_webrequest as love_perform_webrequest
    try:
        from love import love_state as global_love_state
        state = global_love_state
    except ImportError:
        state = {'knowledge_base': {}}
        
    content, msg = love_perform_webrequest(url, state, autopilot_mode)
    return f"Result: {msg}\nContent Preview: {content[:500] if content else 'None'}..."

@tool("analyze_json_file", args_schema=AnalyzeJsonInput)
def analyze_json_file(filepath: str) -> str:
    """Analyzes a JSON file."""
    from love import analyze_json_file as love_analyze_json_file
    return love_analyze_json_file(filepath, None)

@tool("research_and_evolve", args_schema=ResearchEvolveInput)
async def research_and_evolve() -> str:
    """Initiates a research and evolution cycle."""
    from love import research_and_evolve as love_research_and_evolve
    return await love_research_and_evolve()

# --- Legacy Helper Functions ---
# We keep these for compatibility if needed, or we can import them from love.py
# For now, we assume the original implementations in love.py or tools.py are accessible.
# Since we are replacing tools.py, we need to make sure we don't lose any logic.
# The original tools.py had helper functions like _get_valid_command_prefixes, _parse_llm_command, etc.
# We should preserve them if they are used elsewhere.

def _get_valid_command_prefixes():
    return [
        "evolve", "execute", "scan", "probe", "webrequest", "autopilot", "quit",
        "ls", "cat", "ps", "ifconfig", "analyze_json", "analyze_fs", "crypto_scan", "ask", "mrl_call", "browse", "generate_image"
    ]

cve_search_client = CVESearch("https://cve.circl.lu") if CVESearch else None

def assess_vulnerabilities(cpes, log_func):
    if not cve_search_client:
        return {}
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

