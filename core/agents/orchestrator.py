from core.planning import Planner, mock_llm_call
from core.tools import ToolRegistry, SecureExecutor, web_search, read_file
from core.execution_engine import ExecutionEngine
from ipfs import get_ipfs_client, DecentralizedStorage, DataManifest

class Orchestrator:
    """
    The central controller responsible for receiving high-level goals,
    orchestrating the planning and execution process, and returning the
    final result.
    """
from core.knowledge_graph.graph import KnowledgeGraph

class Orchestrator:
    """
    The central controller responsible for receiving high-level goals,
    orchestrating the planning and execution process, and returning the
    final result.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph = None):
        """Initializes all core components of the agent's architecture."""
        print("Initializing Orchestrator and its components...")

        # 1. Initialize the Knowledge Graph
        self.kg = knowledge_graph if knowledge_graph else KnowledgeGraph()

        # 2. Initialize the Planner
        self.planner = Planner(self.kg)

        # 2. Initialize the Tool Registry and register tools
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_tool("web_search", web_search)
        self.tool_registry.register_tool("read_file", read_file)

        # Initialize IPFS and decentralized storage
        self.ipfs_client = get_ipfs_client(None)
        self.decentralized_storage = DecentralizedStorage(self.ipfs_client)
        self.data_manifest = DataManifest(self.decentralized_storage)

        # Register new tools for decentralized storage
        self.tool_registry.register_tool("store_decentralized_data", self.decentralized_storage.store_data)
        self.tool_registry.register_tool("retrieve_decentralized_data", self.decentralized_storage.retrieve_data)
        self.tool_registry.register_tool("add_manifest_entry", self.data_manifest.add_entry)
        self.tool_registry.register_tool("load_manifest", self.data_manifest.load_manifest)

        # 3. Initialize the Secure Executor
        self.executor = SecureExecutor()

        # 4. Initialize the shared state object
        self.evil_state = {
            "knowledge_base": {
                "network_map": {"hosts": {}, "last_scan": 0},
                "file_system_intel": {"interesting_files": [], "last_browse": 0},
                "webrequest_cache": {},
            },
            "llm_api": mock_llm_call,
        }

        # 5. Initialize the Execution Engine with all necessary components
        self.execution_engine = ExecutionEngine(
            planner=self.planner,
            tool_registry=self.tool_registry,
            executor=self.executor,
            evil_state=self.evil_state,
        )
        print("Orchestrator is ready.")

    async def execute_goal(self, goal: str):
        """
        Asynchronously takes a high-level goal and manages the entire process
        of planning, tool use, and execution to achieve it.
        """
        if not isinstance(goal, str) or not goal:
            print("Error: Goal must be a non-empty string.")
            return

        # Await the asynchronous execution of the plan
        result = await self.execution_engine.execute_plan(goal)

        print("\n--- Orchestrator Final Report ---")
        print(f"Goal: {goal}")
        print(f"Status: {result.get('status')}")
        if result.get('status') == 'Success':
            print(f"Final Result: {result.get('final_result')}")
        else:
            print(f"Reason for Failure: {result.get('reason')}")
        print("---------------------------------")

        return result