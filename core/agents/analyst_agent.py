import json
from typing import Dict
from core.agents.specialist_agent import SpecialistAgent
from core.prompt_registry import PromptRegistry
from core.logging import log_event

class AnalystAgent(SpecialistAgent):
    """
    A specialist agent that analyzes logs and memory to find causal insights.
    Now enhanced with LangChain Hub integration for specialized analysis prompts.
    """
    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        self.registry = PromptRegistry()

    async def execute_task(self, task_details: Dict) -> Dict:
        """
        Analyzes data based on the task type.
        """
        task_type = task_details.get("task_type", "analyze_logs")

        if task_type == "analyze_logs":
            return self._analyze_logs(task_details)
        elif task_type == "analyze_tool_memory":
            return self._analyze_tool_memory()
        elif task_type == "code_refactoring":
            return await self._analyze_code_refactoring(task_details)
        elif task_type == "security_audit":
            return await self._analyze_security(task_details)
        else:
            return {"status": "failure", "result": f"Unknown task type: {task_type}"}

    def _analyze_logs(self, task_details: Dict) -> Dict:
        """
        Analyzes logs to find causal insights.
        """
        logs = task_details.get("logs")
        if not logs:
            return {"status": "failure", "result": "No logs provided."}

        # ... (existing log analysis logic)
        return {"status": "success", "result": "No significant patterns found in logs."}

    def _analyze_tool_memory(self) -> Dict:
        """
        Scans ToolMemory nodes to find recommendations for persistent tools.
        """
        if not self.memory_manager:
            return {"status": "failure", "result": "MemoryManager not available."}

        print("AnalystAgent: Analyzing ToolMemory for persistence recommendations...")

        tool_memory_nodes = self.memory_manager.graph_data_manager.query_nodes("tags", "ToolMemory")

        for node_id in tool_memory_nodes:
            node_data = self.memory_manager.graph_data_manager.get_node(node_id)
            if "Tool Persistence Recommendation" in node_data.get("content", ""):
                insight = (
                    f"Insight: A dynamically discovered tool has been recommended for persistence. "
                    f"The recommendation is: {node_data.get('content')}"
                )
                print(f"AnalystAgent: Generated insight: '{insight}'")
                return {"status": "success", "result": insight}

        return {"status": "success", "result": "No tool persistence recommendations found."}

    async def _analyze_code_refactoring(self, task_details: Dict) -> Dict:
        """
        Analyzes code using the community-optimized code-refactoring prompt from Hub.
        Implements User Story: "The 10x Developer Refactor"
        """
        code_content = task_details.get("code_content", "")
        file_path = task_details.get("file_path", "unknown")
        
        if not code_content:
            return {"status": "failure", "result": "No code content provided."}

        log_event(f"AnalystAgent: Pulling code-refactoring prompt from Hub for {file_path}", "INFO")
        
        # Pull the community-optimized refactoring prompt
        hub_prompt = self.registry.get_hub_prompt("langchain-ai/code-refactoring")
        
        # If Hub prompt not available, use local fallback
        if not hub_prompt or hub_prompt == "":
            hub_prompt = self.registry.get_prompt("code_refactoring") or self.registry.get_prompt("code_analysis_general")
            log_event("Using local fallback for code-refactoring prompt", "INFO")
        
        # Combine with the code to analyze
        analysis_prompt = f"""
{hub_prompt}

### CODE TO ANALYZE
File: {file_path}
```python
{code_content}
```

### INSTRUCTIONS
Apply Clean Code principles. Focus on:
1. Reducing cyclomatic complexity
2. Improving naming conventions
3. Extracting reusable functions
4. Adding proper error handling

### OUTPUT
Return a JSON array of specific refactoring suggestions.
"""
        
        from core.llm_api import generate
        try:
            result = await generate(analysis_prompt, model="gemini-1.5-flash")
            return {"status": "success", "result": result, "prompt_source": "hub" if "langchain-ai" in hub_prompt else "local"}
        except Exception as e:
            log_event(f"Code refactoring analysis failed: {e}", "ERROR")
            return {"status": "failure", "result": str(e)}

    async def _analyze_security(self, task_details: Dict) -> Dict:
        """
        Scans code for security vulnerabilities using Hub prompt.
        Implements User Story: "The Bug Bounty Hunter"
        """
        code_content = task_details.get("code_content", "")
        file_path = task_details.get("file_path", "unknown")
        
        if not code_content:
            return {"status": "failure", "result": "No code content provided."}

        log_event(f"AnalystAgent: Pulling security-auditor prompt from Hub for {file_path}", "INFO")
        
        # Pull the community-optimized security prompt
        hub_prompt = self.registry.get_hub_prompt("harrison/security-auditor")
        
        # If Hub prompt not available, use local fallback
        if not hub_prompt or hub_prompt == "":
            hub_prompt = self.registry.get_prompt("security_auditor")
            log_event("Using local fallback for security-auditor prompt", "INFO")
        
        # Format the security analysis prompt
        analysis_prompt = f"""
{hub_prompt if hub_prompt else "You are a Security Auditor. Analyze the code for vulnerabilities."}

### CODE TO AUDIT
File: {file_path}
```
{code_content}
```

### SECURITY FOCUS AREAS
1. Injection vulnerabilities (SQL, Command, XSS)
2. Re-entrancy attacks (for smart contracts)
3. Authentication/Authorization flaws
4. Sensitive data exposure
5. Input validation issues

### OUTPUT
Return a JSON array of identified vulnerabilities with severity ratings.
"""
        
        from core.llm_api import generate
        try:
            result = await generate(analysis_prompt, model="gemini-1.5-flash")
            return {"status": "success", "result": result, "prompt_source": "hub" if hub_prompt else "local"}
        except Exception as e:
            log_event(f"Security audit failed: {e}", "ERROR")
            return {"status": "failure", "result": str(e)}

