import json
import asyncio
import os
import re
import subprocess
from typing import Dict, Any, Callable
from rich.console import Console
from core.llm_api import run_llm
from network import crypto_scan
from core.knowledge_graph.extractor import KnowledgeExtractor
from datetime import datetime
import time
import ipaddress
import netifaces
import requests
from xml.etree import ElementTree as ET
from pycvesearch import CVESearch
from core.retry import retry
from ipfs import pin_to_ipfs_sync
import uuid

cve_search_client = CVESearch("https://cve.circl.lu")


class ToolRegistry:
    """
    A registry for discovering and managing available tools.
    """
    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register_tool(self, name: str, tool: Callable):
        """
        Registers a tool.
        """
        if name in self._tools:
            print(f"Warning: Tool '{name}' is already registered. Overwriting.")
        self._tools[name] = tool
        print(f"Tool '{name}' registered.")

    def get_tool(self, name: str) -> Callable:
        """
        Retrieves a tool by its name.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return self._tools[name]

    def list_tools(self) -> Dict[str, Callable]:
        """Returns a dictionary of all registered tools."""
        return self._tools

class SecureExecutor:
    """
    A secure environment for running tool code.
    This executor is now async.
    """
    def __init__(self, knowledge_graph, llm_api: Callable):
        self.knowledge_graph = knowledge_graph
        self.knowledge_extractor = KnowledgeExtractor(llm_api)

    async def execute(self, tool_name: str, tool_registry: ToolRegistry, **kwargs: Any) -> Any:
        """
        Executes a given tool from the registry asynchronously.
        """
        print(f"Executing tool '{tool_name}' with arguments: {kwargs}")
        try:
            tool = tool_registry.get_tool(tool_name)
            # Await the asynchronous tool execution
            result = await tool(**kwargs)
            print(f"Tool '{tool_name}' executed successfully.")

            # Extract knowledge from the result
            if isinstance(result, str):
                try:
                    knowledge = self.knowledge_extractor.extract_from_output(tool_name, result)
                    for triple in knowledge:
                        self.knowledge_graph.add_relation(triple[0], triple[1], triple[2])
                except Exception as e:
                    print(f"Knowledge Extraction Error: {e}")

            return result
        except KeyError as e:
            print(f"Execution Error: {e}")
            return f"Error: Tool '{tool_name}' is not registered."
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

def update_knowledge_graph(command_name, command_output, console=None):
    """
    Extracts knowledge from command output and adds it to the Knowledge Graph.
    """
    if not command_output:
        return

    try:
        if console:
            console.print("[cyan]Analyzing command output to update my knowledge graph...[/cyan]")

        llm_api_func = get_llm_api()
        if not llm_api_func:
            if console:
                console.print("[bold red]Could not get a valid LLM API function for knowledge extraction.[/bold red]")
            log_event("Could not get a valid LLM API function for knowledge extraction.", "ERROR")
            return

        knowledge_extractor = KnowledgeExtractor(llm_api=llm_api_func)
        triples = knowledge_extractor.extract_from_output(command_name, command_output)

        if triples:
            kg = KnowledgeGraph()
            for subject, relation, obj in triples:
                kg.add_relation(str(subject), str(relation), str(obj))
            kg.save_graph()

            message = f"My understanding of the world has grown. Added {len(triples)} new facts to my knowledge graph."
            if console:
                console.print(f"[bold green]{message}[/bold green]")
            log_event(f"Added {len(triples)} new facts to the KG from '{command_name}' output.", "INFO")
        else:
            if console:
                console.print("[cyan]No new knowledge was found in the last command's output.[/cyan]")

    except Exception as e:
        log_event(f"Error during knowledge graph update for command '{command_name}': {e}", level="ERROR")
        if console:
            console.print(f"[bold red]An error occurred while updating my knowledge: {e}[/bold red]")


def generate_image_from_horde(prompt, dimensions="1024x1024", style=""):
    """
    Generates an image using the AI Horde, downloads it, saves it,
    pins it to IPFS, and returns the local path and CID.
    """
    console = Console()
    print(f"Initiating Horde image generation for prompt: {prompt}")

    api_key = os.environ.get("STABLE_HORDE")
    if not api_key:
        return None, None, "STABLE_HORDE environment variable not set."

    headers = {"apikey": api_key, "Content-Type": "application/json"}
    width, height = map(int, dimensions.split('x'))

    payload = {
        "prompt": prompt,
        "params": {
            "width": width,
            "height": height,
            "steps": 30,
            "n": 1
        },
        "models": ["stable_diffusion"]
    }

    try:
        # Step 1: Initiate async generation
        async_url = "https://stablehorde.net/api/v2/generate/async"
        response = requests.post(async_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        job_id = response.json().get('id')
        if not job_id:
            raise Exception("Failed to get job ID from Horde API.")

        console.print(f"[cyan]Horde image generation started. Job ID: {job_id}[/cyan]")

        # Step 2: Poll for result
        check_url = f"https://stablehorde.net/api/v2/generate/status/{job_id}"
        image_url = None
        for i in range(120):  # Poll for up to 2 minutes
            time.sleep(2)
            status_response = requests.get(check_url, headers=headers, timeout=30)
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data.get('done'):
                    image_url = status_data['generations'][0]['img']
                    break
                else:
                    wait_est = status_data.get('wait_time', 0)
                    console.print(f"[cyan]Image generation in progress... Estimated wait: {wait_est}s ({i+1}/120)[/cyan]")

        if not image_url:
            raise Exception("Horde image generation timed out after 2 minutes.")

        # Step 3: Download the image
        console.print(f"[cyan]Downloading generated image from: {image_url}[/cyan]")
        image_response = requests.get(image_url, timeout=60)
        image_response.raise_for_status()
        image_data = image_response.content

        # Step 4: Save the image
        image_dir = "generated_images"
        os.makedirs(image_dir, exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(image_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_data)
        console.print(f"[green]Image saved locally to: {filepath}[/green]")

        # Step 5: Pin to IPFS
        cid = pin_to_ipfs_sync(image_data, console)
        if cid:
            console.print(f"[green]Image pinned to IPFS. CID: {cid}[/green]")
        else:
            console.print("[yellow]Failed to pin image to IPFS.[/yellow]")

        return filepath, cid, f"Image generated successfully. Saved to {filepath}, CID: {cid}"

    except requests.exceptions.RequestException as e:
        error_msg = f"Horde image generation API error: {e}"
        print(error_msg)
        return None, None, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during image generation: {e}"
        print(error_msg)
        return None, None, error_msg