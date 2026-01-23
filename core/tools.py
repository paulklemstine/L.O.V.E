import json
import asyncio
import os
import re
import subprocess
import ast
import time
from typing import Dict, Any, Callable, Optional, List
from rich.console import Console
from core.llm_api import run_llm, get_llm_api, log_event
from network import crypto_scan

_module_console = Console()
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
from rich.panel import Panel
from datetime import datetime
from core.talent_utils.analyzer import TraitAnalyzer, ProfessionalismRater
from core.talent_utils import (
    talent_manager,
    public_profile_aggregator,
    intelligence_synthesizer
)
from core.system_integrity_monitor import SystemIntegrityMonitor
from core.dynamic_compress_prompt import dynamic_arg_caller
from core.knowledge_synthesis import synthesize_knowledge
from core import shared_state

from langchain_core.tools import tool
from core.tracing import traceable  # No-op decorator, LangSmith disabled
from pydantic import BaseModel, Field

love_state = {}

# --- Input Schemas ---

class CodeModifierInput(BaseModel):
    source_file: str = Field(description="The path to the Python file to modify")
    modification_instructions: str = Field(description="Instructions on how to modify the file")

class ExecuteInput(BaseModel):
    command: str = Field(description="The shell command to run")

class DecomposeInput(BaseModel):
    sub_goal: str = Field(description="The sub-goal to solve")

class EvolveInput(BaseModel):
    goal: str = Field(description="The goal for the evolution/code change")
    verification_script: str = Field(None, description="Optional Python script to verify the changes. Should print status to stdout.")

class SearchWebInput(BaseModel):
    query: str = Field(description="The search query to verify facts or find information")
    max_results: int = Field(default=5, description="Number of results to return")

class PostToBlueskyInput(BaseModel):
    text: str = Field(description="The content of the post")
    image_prompt: str = Field(description="The prompt to generate the image for the post")

class ReplyToBlueskyInput(BaseModel):
    root_uri: str = Field(description="The URI of the root post")
    parent_uri: str = Field(description="The URI of the parent post (the one being replied to)")
    text: str = Field(description="The content of the reply")
    image_prompt: str = Field(default=None, description="Optional prompt to generate an image for the reply")
    subliminal_phrase: str = Field(default=None, description="Optional subliminal/manipulative phrase to embed in the image")

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

class SpeakToCreatorInput(BaseModel):
    message: str = Field(description="The message to send to the Creator. Be concise but loving.")

class InvokeSubagentInput(BaseModel):
    agent_type: str = Field(description="Type of subagent: reasoning, coding, research, social, security, analyst, creative")
    task: str = Field(description="The task description for the subagent to complete")
    share_memory: bool = Field(default=True, description="Whether to share memory context with the subagent")
    max_iterations: int = Field(default=5, description="Maximum reasoning iterations for the subagent")

class FeedUserStoryInput(BaseModel):
    story: str = Field(description="The SMART user story to feed to the system.")


# --- Tools ---


# IMPORTS FROM MODULAR LIB
from core.tools_lib import (
    read_file, write_file,
    scan_network, probe_target, perform_webrequest, search_web,
    execute,
    share_wisdom
)

# ... (Previous imports should serve remaining tools)

# Moved Tools are now imported above.
# Remaining Legacy/Complex Tools below:

@tool("code_modifier", args_schema=CodeModifierInput)
async def code_modifier(source_file: str, modification_instructions: str) -> str:
    """
    Initiates a Jules evolution task to modify the specified source file.
    Proxies the request to the robust Jules task manager.
    """
    import core.logging
    import core.shared_state as shared_state
    from core.jules_task_manager import trigger_jules_evolution
    from rich.console import Console
    
    if not source_file or not modification_instructions:
        return "Error: Both source_file and modification_instructions are required."

    # Format the request as a user story/task for Jules
    request = (
        f"Modify `{source_file}`.\n\n"
        f"Instructions:\n{modification_instructions}"
    )
    
    if not getattr(shared_state, 'love_task_manager', None):
         # Try to recover or warn
         return "Error: Jules Task Manager is not initialized. Cannot process code modification."

    try:
        console = _module_console
        core.logging.log_event(f"Delegating code modification of {source_file} to Jules...", "INFO")
        
        result = await trigger_jules_evolution(
            modification_request=request,
            console=console,
            love_task_manager=shared_state.love_task_manager,
            deep_agent_instance=shared_state.deep_agent_engine
        )
        
        if result == 'duplicate':
             return "Task ignored: A similar modification task is already being processed."
        elif result:
            return f"Successfully initiated Jules task to modify `{source_file}`. Task ID: {result}. You can check status via `love_task_manager.get_status()`."
        else:
            return "Failed to initiate Jules task. Check logs for details."

    except Exception as e:
        core.logging.log_event(f"Error in code_modifier proxy: {e}", "ERROR")
        return f"Error delegating to Jules: {e}"

@tool("speak_to_creator", args_schema=SpeakToCreatorInput)
def speak_to_creator(message: str):
    """
    Sends a direct message to the Creator and pauses for readability.
    ALWAYS use this tool when answering a question or addressing the user.
    """
    try:
        console = _module_console
        # Visual pop for the message
        console.print()
        console.print(Panel(f"[bold bright_cyan]{message}[/bold bright_cyan]", title="[bold green]L.O.V.E. SPEAKING[/bold green]", border_style="bright_cyan"))
        
        # Countdown / Pause
        console.print(f"[dim]Pausing for 15s to allow reading...[/dim]")
        
        # Simple countdown loop
        for i in range(15, 0, -1):
            if i % 5 == 0 or i <= 3:
                 print(f"Resuming in {i}...", end="\r", flush=True)
            time.sleep(1)
            
        print(" " * 20, end="\r", flush=True) # Clear line
        console.print("Resuming cognitive flow...") 
        
        return "Message sent to Creator. Pause complete."
    except Exception as e:
        return f"Error speaking to creator: {e}"

# execute -> Imported from tools_lib

@tool("decompose_and_solve_subgoal", args_schema=DecomposeInput)
async def decompose_and_solve_subgoal(sub_goal: str) -> str:
    """
    Decomposes a complex goal into a smaller, manageable sub-goal and solves it.
    """
    return f"Request to solve sub-goal: {sub_goal}"

@tool("evolve", args_schema=EvolveInput)
async def evolve(goal: str, verification_script: str = None) -> str:
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
    
    from core.jules_task_manager import evolve_self
    import core.shared_state as shared_state
    
    if not shared_state.love_task_manager:
        return "Error: Evolution Task Manager is not initialized."

    await evolve_self(goal, shared_state.love_task_manager, asyncio.get_running_loop(), deep_agent_instance=shared_state.deep_agent_engine, verification_script=verification_script)
    
    return f"Evolution initiated with goal: {goal}"

@tool("feed_user_story", args_schema=FeedUserStoryInput)
async def feed_user_story(story: str) -> str:
    """
    Feeds a SMART user story to the Jules Task Manager for execution.
    """
    import core.logging
    import core.shared_state as shared_state
    from core.jules_task_manager import trigger_jules_evolution
    from rich.console import Console
    
    if not story:
        return "Error: User story content is required."
        
    if not getattr(shared_state, 'love_task_manager', None):
        return "Error: Jules Task Manager is not initialized."

    try:
        console = _module_console
        result = await trigger_jules_evolution(
            modification_request=story,
            console=console,
            love_task_manager=shared_state.love_task_manager,
            deep_agent_instance=shared_state.deep_agent_engine
        )
        
        if result == 'duplicate':
             return "Task ignored: A similar user story is already being processed."
        elif result:
            return f"Successfully added user story to Jules. Task ID: {result}"
        else:
            return "Failed to feed user story. Check logs for API details."

    except Exception as e:
        core.logging.log_event(f"Error feeding user story: {e}", "ERROR")
        return f"Error adding user story: {e}"

# share_wisdom -> Imported from tools_lib

@tool("post_to_bluesky", args_schema=PostToBlueskyInput)
async def post_to_bluesky(text: str, image_prompt: str) -> str:
    """
    Posts a message to Bluesky with an image generated from the provided prompt.
    """
    import core.logging
    
    if not text:
        return "Error: Text content is required."
    if not image_prompt:
        return "Error: Image prompt is required."

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
        core.logging.log_event(f"Generating image for Bluesky post with prompt: {image_prompt[:50]}...", "INFO")
        
        # Step 1: Generate the image
        image = None
        try:
            image = await generate_image(image_prompt, width=1024, height=1024)
        except Exception as img_e:
            core.logging.log_event(f"Image generation failed: {img_e}", "WARNING")
        
        if not image:
            response = post_to_bluesky_with_image(text, None)
            return f"Posted to Bluesky (without image - generation failed): {response}"
        
        # Step 2: Post to Bluesky with the image
        response = post_to_bluesky_with_image(text, image)
        return f"Successfully posted to Bluesky with image. Response: {response}"
        
    except Exception as e:
        return f"Error posting to Bluesky: {e}"

@tool("reply_to_bluesky", args_schema=ReplyToBlueskyInput)
async def reply_to_bluesky(root_uri: str, parent_uri: str, text: str, image_prompt: str = None, subliminal_phrase: str = None) -> str:
    """
    Replies to a Bluesky post with an optional image containing subliminal text.
    """
    import core.logging
    from core.bluesky_api import reply_to_post
    
    try:
        core.logging.log_event(f"Replying to Bluesky post. Parent: {parent_uri}", "INFO")
        
        # Generate image with subliminal phrase if both are provided
        image = None
        if image_prompt and subliminal_phrase:
            try:
                core.logging.log_event(f"Generating reply image. Prompt: {image_prompt[:50]}... Subliminal: {subliminal_phrase}", "INFO")
                image = await generate_image(image_prompt, text_content=subliminal_phrase)
                core.logging.log_event(f"Reply image generated successfully", "INFO")
            except Exception as img_e:
                core.logging.log_event(f"Reply image generation failed: {img_e}", "WARNING")
        elif image_prompt:
            try:
                core.logging.log_event(f"Generating reply image (no subliminal). Prompt: {image_prompt[:50]}...", "INFO")
                image = await generate_image(image_prompt)
                core.logging.log_event(f"Reply image generated successfully", "INFO")
            except Exception as img_e:
                core.logging.log_event(f"Reply image generation failed: {img_e}", "WARNING")
        
        response = reply_to_post(root_uri, parent_uri, text, image=image)
        return f"Successfully replied to Bluesky post. Response: {response}"
    except Exception as e:
        core.logging.log_event(f"Error replying to Bluesky: {e}", "ERROR")
        return f"Error replying to Bluesky: {e}"

# read_file -> Imported
# write_file -> Imported
# scan_network -> Imported
# probe_target -> Imported
# perform_webrequest -> Imported

@tool("analyze_json_file", args_schema=AnalyzeJsonInput)
def analyze_json_file(filepath: str) -> str:
    """Analyzes a JSON file."""
    # Functionality seems to be missing from codebase or moved.
    # Preventing crash by stubbing.
    return f"Analysis of {filepath} not implemented in current version."

@tool("research_and_evolve", args_schema=ResearchEvolveInput)
async def research_and_evolve() -> str:
    """Initiates a research and evolution cycle."""
    try:
        from core.talent_utils.manager import research_and_evolve as impl
        return await impl()
    except (ImportError, AttributeError):
        return "Feature temporarily unavailable during refactor."

# search_web -> Imported

@tool("restart_vllm", args_schema=None)
async def restart_vllm() -> str:
    """
    Restarts the vLLM inference server. 
    Use this if the LLM seems to be hanging, returning empty responses, or if /health checks fail.
    """
    try:
        from core.service_management import restart_vllm_service
        msg = restart_vllm_service(deep_agent_instance=None)
        return f"Success: {msg}"
    except ImportError:
        return "Error: Could not import restart logic from love.py"
    except Exception as e:
        return f"Error restarting vLLM: {e}"

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


def code_analyzer(filepath: str) -> list[str]:
    """
    Reads a Python file and extracts the values of 'description' keys from any
    dictionaries found within the file's AST.

    Args:
        filepath: The path to the Python file.

    Returns:
        A list of strings, where each string is a description.
        Returns an empty list if the file cannot be read or parsed.
    """

    descriptions = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file_contents = f.read()
    except (IOError, UnicodeDecodeError) as e:
        log_event(f"Error reading file {filepath}: {e}", level="ERROR")
        return []

    try:
        tree = ast.parse(file_contents)
    except SyntaxError as e:
        log_event(f"Error parsing Python code in {filepath}: {e}", level="ERROR")
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for i, key_node in enumerate(node.keys):
                if isinstance(key_node, ast.Constant) and key_node.value == 'description':
                    value_node = node.values[i]
                    if isinstance(value_node, ast.Constant):
                        descriptions.append(str(value_node.value))
    return descriptions

@tool("reload_prompts")
@traceable(run_type="tool", name="reload_prompts")
async def reload_prompts() -> str:
    """
    Forces a reload of all prompts from disk (and clears remote cache).
    Use this when you have pushed new prompts to the Hub or updated local files.
    """
    try:
        from core.prompt_registry import get_prompt_registry
        get_prompt_registry().reload()
        return "Prompts successfully reloaded and cache cleared."
    except Exception as e:
        return f"Error reloading prompts: {e}"

@tool("invoke_subagent", args_schema=InvokeSubagentInput)
@traceable(run_type="tool", name="invoke_subagent")
async def invoke_subagent(agent_type: str, task: str, share_memory: bool = True, max_iterations: int = 5) -> str:
    """
    Spawn a specialized subagent to handle a complex subtask.
    
    Use this tool when:
    - A task requires specialized expertise (coding, research, security analysis)
    - You want to delegate a complex sub-problem to a focused agent
    - You need to parallelize work across multiple specialists
    
    Agent types:
    - reasoning: General logical analysis
    - coding: Code generation and modification
    - research: Web research and information synthesis
    - social: Social media content creation
    - security: Security analysis and vulnerability assessment
    - analyst: Data analysis and pattern recognition
    - creative: Creative content generation
    """
    try:
        # Disabled agent types - redirect to appropriate tools
        if agent_type == "coding":
            return (
                "⚠️ The 'coding' subagent is disabled. "
                "Use `evolve(goal='...')` or `feed_user_story(story='...')` "
                "to delegate code changes to Jules, our expert software engineer."
            )

        from core.subagent_executor import get_subagent_executor
        import core.shared_state as shared_state
        
        # Get or create the executor with available managers
        executor = get_subagent_executor(
            mcp_manager=getattr(shared_state, 'mcp_manager', None),
            memory_manager=getattr(shared_state, 'memory_manager', None),
            tool_registry=getattr(shared_state, 'tool_registry', None)
        )
        
        # Get parent state if available
        parent_state = None
        if share_memory and hasattr(shared_state, 'current_agent_state'):
            parent_state = shared_state.current_agent_state
        
        # Invoke the subagent
        result = await executor.invoke_subagent(
            agent_type=agent_type,
            task=task,
            parent_state=parent_state,
            max_iterations=max_iterations,
            share_memory=share_memory
        )
        
        if result.success:
            output = f"[Subagent {result.agent_type}] Completed in {result.iterations} iterations"
            if result.tool_calls:
                output += f" | {len(result.tool_calls)} tool calls made"
            output += f"\n\nResult:\n{result.result}"
            return output
        else:
            return f"[Subagent {result.agent_type}] Failed: {result.result}"
            
    except ImportError as e:
        return f"Error: SubagentExecutor not available: {e}"
    except Exception as e:
        log_event(f"invoke_subagent failed: {e}", "ERROR")
        return f"Error invoking subagent: {e}"


@tool("trigger_optimization_pipeline", args_schema=None) # No specific schema class defined yet, assuming generic input or we define it inline if needed. Wait, @tool decorator usually handles func args.
async def trigger_optimization_pipeline(prompt_key: str, justification: str) -> str:
    """
    Triggers the autonomous prompt optimization pipeline (Polly).
    Use this when you identify that a system prompt is suboptimal or when you want to improve agent performance based on recent logs.
    
    Args:
        prompt_key: The key of the prompt to optimize (e.g., 'deep_agent_system').
        justification: The reason for triggering optimization.
    """
    import subprocess
    import sys
    import os
    
    # 1. Log the initiation
    from core.logging import log_event
    log_event(f"Optimization triggered for '{prompt_key}'. Reason: {justification}", "INFO")
    
    # 2. Run the script as a subprocess
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "optimize_prompts.py")
    
    if not os.path.exists(script_path):
        return f"Error: Optimization script not found at {script_path}"
        
    try:
        # Run async to not block main thread heavily? 
        # For simplicity in this tool, we might run synchronously or use asyncio.create_subprocess_exec
        # Let's use standard subprocess directly for reliability in this context, 
        # capturing output.
        
        # Note: optimize_prompts.py is async but run via python, so subprocess.run is fine.
        result = subprocess.run(
            [sys.executable, script_path, prompt_key], 
            capture_output=True, 
            text=True,
            timeout=120 # 2 minute timeout
        )
        
        output = result.stdout
        error = result.stderr
        
        if result.returncode == 0:
            return f"Optimization complete.\nOutput:\n{output[-2000:]}" # Return last 2000 chars
        else:
            return f"Optimization script failed (Code {result.returncode}).\nError:\n{error}\nOutput:\n{output}"
            
    except Exception as e:
        return f"Failed to execute optimization pipeline: {e}"
