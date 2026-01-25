
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import core.shared_state as shared_state

class ScanNetworkInput(BaseModel):
    autopilot_mode: bool = Field(default=False, description="Whether to run in autopilot mode")

class ProbeTargetInput(BaseModel):
    ip_address: str = Field(description="The IP address to probe")
    autopilot_mode: bool = Field(default=False, description="Whether to run in autopilot mode")

class WebRequestInput(BaseModel):
    url: str = Field(description="The URL to fetch")
    autopilot_mode: bool = Field(default=False, description="Whether to run in autopilot mode")

class SearchWebInput(BaseModel):
    query: str = Field(description="The search query to verify facts or find information")
    max_results: int = Field(default=5, description="Number of results to return")


@tool("scan_network", args_schema=ScanNetworkInput)
def scan_network(autopilot_mode: bool = False) -> str:
    """Scans the local network for active hosts."""
    from network import scan_network as love_scan_network
    
    # Ensure knowledge base is available
    if not hasattr(shared_state, 'knowledge_base'):
         return "Error: Knowledge Base not initialized."

    ips, log = love_scan_network(shared_state.knowledge_base, autopilot_mode)
    return f"Found IPs: {ips}\nLog: {log}"

@tool("probe_target", args_schema=ProbeTargetInput)
def probe_target(ip_address: str, autopilot_mode: bool = False) -> str:
    """Performs a deep probe on a single IP address."""
    from network import probe_target as love_probe_target
    
    if not hasattr(shared_state, 'knowledge_base'):
         return "Error: Knowledge Base not initialized."
        
    ports, output = love_probe_target(ip_address, shared_state.knowledge_base, autopilot_mode)
    return f"Probe Output: {output}"

@tool("perform_webrequest", args_schema=WebRequestInput)
def perform_webrequest(url: str, autopilot_mode: bool = False) -> str:
    """Fetches the content of a URL."""
    from network import perform_webrequest as love_perform_webrequest
    
    if not hasattr(shared_state, 'knowledge_base'):
         return "Error: Knowledge Base not initialized."
        
    content, msg = love_perform_webrequest(url, shared_state.knowledge_base, autopilot_mode)
    return f"Result: {msg}\nContent Preview: {content[:500] if content else 'None'}..."

@tool("search_web", args_schema=SearchWebInput)
def search_web(query: str, max_results: int = 5) -> str:
    """
    Searches the web for information using DuckDuckGo.
    Use this to verify facts or retrieve up-to-date knowledge.
    """
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            # Use text search
            for r in ddgs.text(query, max_results=max_results):
                results.append(r)
        
        if not results:
            return "No results found."
            
        formatted_results = []
        for i, res in enumerate(results):
            formatted_results.append(f"[{i+1}] {res.get('title', 'No Title')}\n    URL: {res.get('href', 'No URL')}\n    Summary: {res.get('body', 'No snippet')}")
            
        return "\n\n".join(formatted_results)
        
    except ImportError:
        return "Error: duckduckgo-search library is not installed."
    except Exception as e:
        return f"Error searching the web: {e}"
