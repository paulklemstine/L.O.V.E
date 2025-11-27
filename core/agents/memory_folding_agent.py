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

        try:
            response = await run_llm(prompt_key="memory_folding_compression", prompt_vars={"full_interaction": full_interaction}, is_source_code=False, force_model=None)
            summary_json = json.loads(response.get("result", "{}"))
            return summary_json
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error processing LLM response for memory compression: {e}")
            return {}

    async def execute_task(self, task_details: Dict) -> Dict:
        """
        Executes the memory folding process.
        """
        print("--- MemoryFoldingAgent: Starting memory folding process ---")
        min_length = task_details.get("min_length", 5) # Use a higher threshold

        chains = self._query_memory_chains(min_length=min_length)

        if not chains:
            return {"status": "success", "result": "No memory chains met the criteria for folding."}

        print(f"Identified {len(chains)} memory chains for folding.")

        folded_count = 0
        for i, chain in enumerate(chains):
            print(f"Processing chain {i+1}/{len(chains)} (length {len(chain)})...")
            summary = await self._compress_chain_with_llm(chain)
            if not summary:
                print(f"  - Failed to compress chain {i+1}.")
                continue

            summary_content = (
                f"Episodic Summary: {summary.get('Episodic Memory', 'N/A')}\n\n"
                f"Final Outcome: {summary.get('Working Memory', 'N/A')}\n\n"
                f"Tool Usage Summary: {summary.get('Tool Memory', 'N/A')}"
            )

            # Use the main memory manager method to store the note.
            # This ensures the new note is also linked to other relevant memories.
            folded_note = await self.memory_manager.add_agentic_memory_note(
                content=summary_content,
                external_tags=["FoldedMemory"]
            )

            if folded_note:
                print(f"  - Stored new FoldedMemory note with ID: {folded_note.id}")
                # Manually add the summarization links
                start_node_id = chain[0]
                end_node_id = chain[-1]

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
                folded_count += 1
            else:
                print(f"  - Failed to store the folded memory for chain {i+1}.")

        result_message = f"Memory folding process complete. Successfully folded {folded_count}/{len(chains)} chains."
        return {"status": "success", "result": result_message}
