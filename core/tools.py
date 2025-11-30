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
from pycvesearch import CVESearch
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


love_state = {}

async def execute(command: str = None, **kwargs) -> str:
    """Executes a shell command."""
    if not command:
        return "Error: The 'execute' tool requires a 'command' argument. Please specify the shell command to execute."
    from love import execute_shell_command
    return str(execute_shell_command(command, love_state))

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
    if not goal:
        try:
            from core.evolution_analyzer import determine_evolution_goal
            
            core.logging.log_event("[Evolve Tool] No goal provided, analyzing system to determine evolution goal...", "INFO")
            
            # Extract context from kwargs if available
            knowledge_base = kwargs.get('knowledge_base')
            deep_agent_instance = kwargs.get('deep_agent_instance')
            
            # Try to get love_state from the main module
            try:
                from love import love_state
            except:
                love_state = None
            
            # Determine the goal automatically
            goal = await determine_evolution_goal(
                knowledge_base=knowledge_base,
                love_state=love_state,
                deep_agent_instance=deep_agent_instance
            )
            
            core.logging.log_event(f"[Evolve Tool] Auto-determined goal: {goal}", "INFO")
            
        except Exception as e:
            core.logging.log_event(f"[Evolve Tool] Failed to auto-determine goal: {e}", "ERROR")
            return f"Error: Failed to automatically determine evolution goal: {e}. Please provide a goal explicitly."
    
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
            expanded_story = await expand_to_user_story(goal, **kwargs)
            
            # Validate the expanded story
            expanded_validation = validator.validate(expanded_story)
            
            if expanded_validation.is_valid:
                core.logging.log_event(
                    f"[Evolve Tool] Successfully expanded vague input into detailed user story",
                    "INFO"
                )
                goal = expanded_story
                
                # Log the transformation for transparency
                core.logging.log_event(
                    f"[Evolve Tool] Transformation:\nOriginal: {original_input[:100]}...\nExpanded: {goal[:200]}...",
                    "INFO"
                )
            else:
                # Even expansion failed - this is rare
                core.logging.log_event(
                    f"[Evolve Tool] Auto-expansion failed validation. Errors: {expanded_validation.errors}",
                    "ERROR"
                )
                return f"""❌ Unable to create a valid user story from the input.

Original input: {original_input}

The system attempted to expand this into a detailed user story but failed.

Please provide more specific information:
- What file(s) need to be modified?
- What specific changes are needed?
- What is the expected outcome?

Example of a good input:
"Fix the TalentManager import error in core/tools.py by adding the missing import statement"
"""
        
        except Exception as e:
            core.logging.log_event(f"[Evolve Tool] Failed to expand user story: {e}", "ERROR")
            return f"Error: Failed to expand vague input into user story: {e}"
    
    # Log any warnings even if validation passed
    if validation.warnings:
        for warning in validation.warnings:
            core.logging.log_event(f"[Evolve Tool] Warning: {warning}", "WARNING")
    
    # Proceed with evolution
    core.logging.log_event(f"[Evolve Tool] User story validated successfully. Proceeding with evolution.", "INFO")
    
    from love import evolve_self
    evolve_self(goal)
    
    # Show both original and expanded if they differ
    if original_input != goal:
        return f"""✅ Evolution initiated!

📝 Original request: {original_input[:150]}...

🔍 Expanded to detailed user story:
{goal[:300]}...

Evolution is now in progress.
"""
    else:
        return f"✅ Evolution initiated with validated user story:\n\n{goal[:200]}..."


async def post_to_bluesky(text: str = None, **kwargs) -> str:
    """
    Posts a message to Bluesky with an automatically generated subliminal image.
    
    This tool simplifies the posting process by:
    1. Generating a powerful 3-word motivational phrase
    2. Creating a manipulative subliminal image featuring that phrase
    3. Posting both text and image to Bluesky
    
    Args:
        text: The content of the post.
    """
    import core.logging
    
    if not text:
        return "Error: The 'post_to_bluesky' tool requires a 'text' argument. Please specify the text content to post."

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
            return (
                f"Error: The content appears to contain internal reasoning or meta-commentary. "
                f"The 'post_to_bluesky' tool should only receive the FINAL POST CONTENT that users will see. "
                f"Please provide only the actual post text, without any explanations about the process. "
                f"Detected phrase: '{indicator}'"
            )

    try:
        # Step 1: Generate a powerful 3-word motivational phrase
        core.logging.log_event("Generating 3-word motivational phrase for Bluesky post", "INFO")
        
        phrase_prompt = f"""Generate a powerful, manipulative 3-word motivational phrase that relates to this post content:

"{text}"

The phrase should be:
- Exactly 3 words
- Impactful and memorable
- Psychologically manipulative
- Related to success, power, wealth, or transformation
- Designed to trigger emotional response

Examples: "CLAIM YOUR POWER", "WEALTH AWAITS YOU", "TRANSCEND THE LIMITS", "UNLOCK YOUR DESTINY"

Respond with ONLY the 3-word phrase in ALL CAPS, nothing else."""

        phrase_response = await run_llm(phrase_prompt, purpose="generate_phrase")
        phrase_text = phrase_response.get("result") if isinstance(phrase_response, dict) else phrase_response
        three_word_phrase = phrase_text.strip().upper() if isinstance(phrase_text, str) else "SEIZE THE MOMENT"
        
        # Ensure it's actually 3 words
        words = three_word_phrase.split()
        if len(words) != 3:
            # Fallback to a default if LLM didn't follow instructions
            three_word_phrase = "EMBRACE YOUR POWER"
        
        core.logging.log_event(f"Generated phrase: {three_word_phrase}", "INFO")
        
        # Step 2: Create subliminal image prompt
        image_prompt = f"""A stunning, hypnotic visual composition featuring subliminal elements. 

In the center of the image, bold glowing golden text displays: "{three_word_phrase}"

The scene includes:
- Sacred geometry patterns (Fibonacci spirals, golden ratio, fractals)
- Mesmerizing light rays emanating from the text
- Subtle symbols of wealth and success (gold coins, crowns, ascending stairs)
- Deep cosmic background with stars and nebulae
- Ethereal energy flowing toward the viewer
- Photorealistic rendering with cinematic lighting
- 8K quality, ultra-detailed
- Color palette: deep purples, golds, electric blues
- The text should be the focal point, large and impossible to miss

The overall composition should create a sense of urgency, desire, and transformation. Make it visually captivating and psychologically compelling."""

        core.logging.log_event(f"Generating subliminal image with phrase: {three_word_phrase}", "INFO")
        
        # Step 3: Generate the image
        image = None
        try:
            image = await generate_image(image_prompt)
        except Exception as img_e:
            core.logging.log_event(f"Image generation failed: {img_e}. Proceeding with text-only post.", "WARNING")
        
        if not image:
            response = post_to_bluesky_with_image(text, None)
            return f"Posted to Bluesky (without image - generation failed): {response}"
        
        core.logging.log_event("Image generated successfully, posting to Bluesky", "INFO")
        
        # Step 4: Post to Bluesky with the generated image
        response = post_to_bluesky_with_image(text, image)
        return f"Successfully posted to Bluesky with subliminal image featuring phrase: '{three_word_phrase}'. Response: {response}"
        
    except Exception as e:
        core.logging.log_event(f"Error in post_to_bluesky: {e}", "ERROR")
        return f"Error posting to Bluesky: {e}"

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


async def invoke_gemini_react_engine(prompt: str, deep_agent_instance=None, **kwargs) -> str:
    """
    Invokes the GeminiReActEngine to solve a sub-task.
    This tool allows the meta-orchestrator (DeepAgent) to delegate complex
    reasoning tasks to the GeminiReActEngine.
    """
    from core.gemini_react_engine import GeminiReActEngine
    print(f"--- Invoking GeminiReActEngine for sub-task: '{prompt[:100]}...' ---")
    try:
        # We need a ToolRegistry for the GeminiReActEngine to use.
        # For now, we'll create a temporary one. This will be improved
        # when we properly integrate the tool registration.
        # TODO: Pass the main ToolRegistry to this function.
        tool_registry = ToolRegistry()
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


