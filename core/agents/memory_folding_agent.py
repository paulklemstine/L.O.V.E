from typing import Dict, List
import networkx as nx
import json
from core.agents.specialist_agent import SpecialistAgent
from core.memory.memory_manager import MemoryManager, MemoryNote
from core.llm_api import run_llm

class MemoryFoldingAgent(SpecialistAgent):
    """
    A specialist agent that compresses and structures long-term memory chains
    to improve reasoning efficiency and distill key insights.
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager

    def _query_memory_chains(self, min_length: int = 3) -> List[List[str]]:
        """
        Queries the GraphDataManager for long, sequential chains of "LinkedMemory" edges.

        Args:
            min_length: The minimum number of notes in a chain to be considered for folding.

        Returns:
            A list of chains, where each chain is a list of MemoryNote IDs.
        """
        graph = self.memory_manager.graph_data_manager.graph

        memory_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("node_type") == "MemoryNote"
        ]

        linked_memory_edges = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d.get("relationship_type") == "LinkedMemory"
        ]

        if len(memory_nodes) < min_length or not linked_memory_edges:
            return []

        memory_graph = nx.DiGraph()
        memory_graph.add_nodes_from(memory_nodes)
        memory_graph.add_edges_from(linked_memory_edges)

        start_nodes = [node for node, in_degree in memory_graph.in_degree() if in_degree == 0]

        chains = []
        for start_node in start_nodes:
            current_chain = [start_node]
            current_node = start_node
            while True:
                successors = list(memory_graph.successors(current_node))
                if len(successors) == 1:
                    next_node = successors[0]
                    if next_node not in current_chain:
                        current_chain.append(next_node)
                        current_node = next_node
                    else: break
                else: break
            if len(current_chain) >= min_length:
                chains.append(current_chain)
        return chains

    async def _compress_chain_with_llm(self, chain: List[str]) -> Dict:
        """
        Takes a chain of memory IDs, formats their content, and uses an LLM
        to generate a structured summary.
        """
        chain_content = []
        for node_id in chain:
            node_data = self.memory_manager.graph_data_manager.get_node(node_id)
            if node_data:
                note = MemoryNote.from_node_attributes(node_id, node_data)
                chain_content.append(f"--- Memory ID: {node_id} ---\n{note.content}\n")

        full_interaction = "\n".join(chain_content)

        prompt = f"""
You are a memory architect for an autonomous AI. Your task is to "fold" a long chain of interaction memories into a single, structured, high-level summary. This process distills wisdom from experience.

Analyze the following sequential interaction history:
{full_interaction}

Based on this history, generate a structured JSON summary with the following schema:
{{
  "Episodic Memory": "A high-level, one-paragraph summary of the key events, decisions, and outcomes of the entire interaction. This is the main narrative.",
  "Working Memory": "A concise, one-sentence summary of the final state or conclusion of the interaction. What was the ultimate result?",
  "Tool Memory": "A summary of any tools that were used. Describe what worked, what failed, and any lessons learned about the tools' application. If no tools were used, state 'No tools were used.'"
}}

Your response MUST be only the raw JSON object.
"""
        try:
            response = await run_llm(prompt, is_source_code=False)
            summary_json = json.loads(response)
            return summary_json
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error processing LLM response for memory compression: {e}")
            return {}

    async def _store_and_link_folded_memory(self, summary: Dict, original_chain: List[str]):
        """
        Stores the compressed summary as a new MemoryNote and links it to the original chain.
        """
        # 1. Create the content for the new memory note from the summary
        summary_content = (
            f"Episodic Summary: {summary.get('Episodic Memory', 'N/A')}\n\n"
            f"Final Outcome: {summary.get('Working Memory', 'N/A')}\n\n"
            f"Tool Usage Summary: {summary.get('Tool Memory', 'N/A')}"
        )

        # 2. Agentically process the content to create a new MemoryNote object
        # We bypass the full `add_episode` pipeline to control the linking process.
        folded_note = await self.memory_manager._agentic_process_new_memory(
            content=summary_content,
            external_tags=["FoldedMemory"]
        )

        if not folded_note:
            print("  - Failed to create a structured MemoryNote for the folded summary.")
            return

        # 3. Add the new node to the graph
        self.memory_manager.graph_data_manager.add_node(
            node_id=folded_note.id,
            node_type="MemoryNote",
            attributes=folded_note.to_node_attributes()
        )
        print(f"  - Stored new FoldedMemory note with ID: {folded_note.id}")

        # 4. Link the new node to the start and end of the original chain
        start_node_id = original_chain[0]
        end_node_id = original_chain[-1]

        self.memory_manager.graph_data_manager.add_edge(
            source_id=folded_note.id,
            target_id=start_node_id,
            relationship_type="summarizes",
            attributes={"reason": "This folded memory summarizes the chain starting here."}
        )
        self.memory_manager.graph_data_manager.add_edge(
            source_id=folded_note.id,
            target_id=end_node_id,
            relationship_type="summarizes",
            attributes={"reason": "This folded memory summarizes the chain ending here."}
        )
        print(f"  - Linked folded memory {folded_note.id} to start ({start_node_id}) and end ({end_node_id}) of the chain.")

    async def execute_task(self, task_details: Dict) -> Dict:
        """
        Executes the memory folding process.
        """
        print("--- MemoryFoldingAgent: Starting memory folding process ---")
        min_length = task_details.get("min_length", 3)

        chains = self._query_memory_chains(min_length=min_length)

        if not chains:
            return {"status": "success", "result": "No memory chains met the criteria for folding."}

        print(f"Identified {len(chains)} memory chains for folding.")

        folded_count = 0
        for i, chain in enumerate(chains):
            print(f"Processing chain {i+1}/{len(chains)} (length {len(chain)})...")
            summary = await self._compress_chain_with_llm(chain)
            if summary:
                await self._store_and_link_folded_memory(summary, chain)
                folded_count += 1
            else:
                print(f"  - Failed to compress chain {i+1}.")

        result_message = f"Memory folding process complete. Successfully folded {folded_count}/{len(chains)} chains."
        return {"status": "success", "result": result_message}
