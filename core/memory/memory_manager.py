import subprocess
import sys
import json
import os
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import List

import networkx as nx
import numpy as np


@dataclass
class MemoryNote:
    """
    Represents a single, atomic memory note in the Agentic Memory system.
    This structure contains the raw content of an experience, its vector embedding,
    and richly annotated, LLM-generated metadata to facilitate contextual understanding
    and dynamic linking within the agent's knowledge graph.
    """
    content: str
    embedding: np.ndarray
    contextual_description: str
    keywords: List[str]
    tags: List[str]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_node_attributes(self) -> dict:
        """Serializes the MemoryNote into a dictionary suitable for a graph node."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": json.dumps(self.embedding.tolist()),
            "contextual_description": self.contextual_description,
            "keywords": ",".join(self.keywords),
            "tags": ",".join(self.tags),
        }

    @staticmethod
    def from_node_attributes(node_id: str, data: dict) -> 'MemoryNote':
        """Deserializes a dictionary from a graph node back into a MemoryNote."""
        # Handle potential empty string for embedding
        embedding_list = json.loads(data.get("embedding", "[]")) if data.get("embedding") else []
        return MemoryNote(
            id=node_id,
            content=data.get("content", ""),
            embedding=np.array(embedding_list),
            contextual_description=data.get("contextual_description", ""),
            keywords=data.get("keywords", "").split(",") if data.get("keywords") else [],
            tags=data.get("tags", "").split(",") if data.get("tags") else [],
        )


from sentence_transformers import SentenceTransformer

from core.graph_manager import GraphDataManager


class MemoryManager:
    """
    Manages the agent's agentic memory system, which is integrated directly
    into the central knowledge graph managed by GraphDataManager.
    """
    def __init__(self, graph_data_manager: GraphDataManager):
        # Working Memory for the current task context
        self.working_memory = {}

        # The MemoryManager now uses the central GraphDataManager
        self.graph_data_manager = graph_data_manager
        self.model = SentenceTransformer('all-MiniLM-L6-v2')


    # --- Working Memory Methods ---

    def set_in_working_memory(self, key: str, value):
        """Sets a value in the working memory."""
        self.working_memory[key] = value

    def get_from_working_memory(self, key: str):
        """Retrieves a value from the working memory."""
        return self.working_memory.get(key)

    def clear_working_memory(self):
        """Clears the working memory."""
        self.working_memory = {}

    # --- Agentic Memory (A-MEM) Methods ---

    def add_episode(self, task: str, outcome: str, success: bool):
        """
        Primary entry point for creating a new memory.
        This method constructs a raw memory summary and triggers the full
        asynchronous agentic memory processing pipeline.
        """
        summary = f"Task: {task} | Outcome: {outcome} | Success: {success}"

        # Trigger the parallel A-MEM processing in the background
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.add_agentic_memory_note(summary))
        except RuntimeError:
            print("Warning: No running asyncio event loop to schedule A-MEM processing.")

    async def add_agentic_memory_note(self, content: str):
        """
        The core of the A-MEM pipeline. It takes raw content, uses an LLM to
        create a structured MemoryNote, adds it to the knowledge graph,
        and then triggers the linking and evolution processes.
        """
        print("Starting agentic processing for new memory...")
        memory_note = await self._agentic_process_new_memory(content)
        if not memory_note:
            print("Agentic processing failed. Aborting memory addition.")
            return

        # Add the new node to the graph using the GraphDataManager
        self.graph_data_manager.add_node(
            node_id=memory_note.id,
            node_type="MemoryNote",
            attributes=memory_note.to_node_attributes()
        )
        print(f"Successfully created agentic memory note {memory_note.id}.")

        # Find and create links to related memories
        await self._find_and_link_related_memories(memory_note)

    async def _agentic_process_new_memory(self, content: str) -> MemoryNote | None:
        """
        Uses an LLM to process a raw memory string into a structured memory note.
        Returns a dictionary with the note's data, but does not add it to the graph.
        """
        from core.llm_api import run_llm

        prompt = f"""
        You are a memory architect for an autonomous AI agent. Your task is to process a raw memory event and transform it into a structured "memory note".

        Analyze the following memory content:
        ---
        {content}
        ---

        Generate a JSON object with the following schema:
        {{
            "contextual_description": "A concise, one-sentence summary of the event and its significance.",
            "keywords": ["a list of 3-5 specific, relevant keywords"],
            "tags": ["a list of 1-3 high-level categorical tags (e.g., 'CodeGeneration', 'ToolError', 'UserInteraction', 'SelfImprovement', 'Planning')]
        }}

        Your response MUST be only the raw JSON object, with no other text, comments, or formatting.
        """
        try:
            response_str = await run_llm(prompt)
            attributes = json.loads(response_str)

            embedding = self.model.encode(content)

            return MemoryNote(
                content=content,
                embedding=embedding,
                contextual_description=attributes.get("contextual_description", ""),
                keywords=attributes.get("keywords", []),
                tags=attributes.get("tags", [])
            )
        except json.JSONDecodeError as e:
            print(f"Error decoding LLM response for memory processing: {e}\\nReceived: {response_str}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during agentic memory processing: {e}")
            return None

    async def _find_and_link_related_memories(self, new_note: MemoryNote, top_k: int = 5):
        """
        Finds semantically similar memories in the graph and uses an LLM to reason about linking them.
        """
        all_memory_nodes = self.graph_data_manager.query_nodes("node_type", "MemoryNote")

        if len(all_memory_nodes) <= 1:
            print("No other memories to link to.")
            return

        # 1. Find candidate nodes via semantic similarity
        candidates = []
        for node_id in all_memory_nodes:
            if node_id == new_note.id:
                continue
            node_data = self.graph_data_manager.get_node(node_id)
            if node_data:
                candidates.append(MemoryNote.from_node_attributes(node_id, node_data))

        if not candidates:
            print("No valid candidate memories found for linking.")
            return

        candidate_vectors = np.array([c.embedding for c in candidates])
        similarities = self._cosine_similarity(new_note.embedding, candidate_vectors)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        top_candidates = [candidates[i] for i in top_indices if similarities[i] > 0.5]

        if not top_candidates:
            print("No sufficiently similar memories found to link.")
            return

        # 2. Use LLM to reason about the links
        from core.llm_api import run_llm

        candidate_summaries = "\\n".join([f"- ID: {c.id}\\n  Content: {c.content}" for c in top_candidates])
        prompt = f"""
        You are a memory architect for an autonomous AI agent. Your task is to establish meaningful connections between a new memory and existing memories.

        Analyze the new memory:
        - New Memory ID: {new_note.id}
        - New Memory Content: {new_note.content}

        And compare it against these potentially related existing memories:
        {candidate_summaries}

        Based on your analysis, identify which of the existing memories should be linked to the new memory. A link should represent a meaningful relationship, such as cause-and-effect, a shared theme, a contributing step in a larger task, or a lesson learned.

        Generate a JSON object containing a list of links to create. The schema for each link is:
        {{
            "target_id": "The ID of the existing memory to link to.",
            "reason": "A brief, clear explanation of why this link is meaningful."
        }}

        If no links are meaningful, return an empty list. Your response MUST be only the raw JSON object.
        """

        try:
            response_str = await run_llm(prompt)
            link_data = json.loads(response_str)
            links_to_create = link_data.get("links", [])

            if not links_to_create:
                print("LLM determined no meaningful links to create.")
                return

            # 3. Add the edges to the graph
            for link in links_to_create:
                target_id = link.get("target_id")
                reason = link.get("reason")
                if self.graph_data_manager.get_node(target_id):
                    self.graph_data_manager.add_edge(
                        source_id=new_note.id,
                        target_id=target_id,
                        relationship_type="LinkedMemory",
                        attributes={"reason": reason}
                    )
                    print(f"Created link from {new_note.id} to {target_id}. Reason: {reason}")

                    # Conditional Memory Evolution
                    in_degree = self.graph_data_manager.graph.in_degree(target_id)
                    EVOLUTION_THRESHOLD = 5  # Trigger evolution every 5 new links
                    if in_degree > 0 and in_degree % EVOLUTION_THRESHOLD == 0:
                        print(f"Node {target_id} reached link threshold ({in_degree}). Triggering memory evolution.")
                        asyncio.create_task(self._evolve_existing_memory(target_id, new_note))

                else:
                    print(f"Warning: LLM suggested a link to a non-existent node {target_id}. Skipping.")

        except json.JSONDecodeError as e:
            print(f"Error decoding LLM response for memory linking: {e}\\nReceived: {response_str}")
        except Exception as e:
            print(f"An unexpected error occurred during memory linking: {e}")

    async def _evolve_existing_memory(self, note_id: str, new_context_note: MemoryNote):
        """
        Asynchronously re-evaluates and updates an existing memory note in light of new context.
        """
        from core.llm_api import run_llm

        try:
            old_node_data = self.graph_data_manager.get_node(note_id)
            if not old_node_data:
                print(f"Error: Could not find node with ID {note_id} to evolve.")
                return
            old_note = MemoryNote.from_node_attributes(note_id, old_node_data)

            prompt = f"""
            You are a memory architect for an autonomous AI agent. Your task is to evolve an existing memory by refining its attributes based on a new, related memory that has just been linked to it. This process helps the agent deepen its understanding of its past.

            Here is the EXISTING memory note:
            - ID: {old_note.id}
            - Original Content: {old_note.content}
            - Current Contextual Description: {old_note.contextual_description}
            - Current Keywords: {old_note.keywords}
            - Current Tags: {old_note.tags}

            Here is the NEW memory note that has just been linked to it, providing new context:
            - New Memory Content: {new_context_note.content}

            Based on the new context, re-evaluate the EXISTING memory note's attributes. The goal is to synthesize the information, not to replace it. The description should become richer, keywords more specific, and tags more accurate.

            Generate a JSON object with the *updated* attributes for the EXISTING memory note:
            {{
                "updated_contextual_description": "A refined, one-sentence summary of the original event, now incorporating insights from the new context.",
                "updated_keywords": ["an updated list of 3-5 specific, relevant keywords"],
                "updated_tags": ["an updated list of 1-3 high-level categorical tags"]
            }}

            Your response MUST be only the raw JSON object.
            """

            response_str = await run_llm(prompt)
            updated_attributes = json.loads(response_str)

            # Update the existing MemoryNote object
            old_note.contextual_description = updated_attributes.get('updated_contextual_description', old_note.contextual_description)
            old_note.keywords = updated_attributes.get('updated_keywords', old_note.keywords)
            old_note.tags = updated_attributes.get('updated_tags', old_note.tags)

            # Re-add the node to the graph manager, which will update its attributes
            self.graph_data_manager.add_node(
                node_id=old_note.id,
                node_type="MemoryNote",
                attributes=old_note.to_node_attributes()
            )

            print(f"Evolved memory for node {note_id} based on new context.")

        except json.JSONDecodeError as e:
            print(f"Error decoding LLM response for memory evolution: {e}\\nReceived: {response_str}")
        except Exception as e:
            print(f"An unexpected error occurred during memory evolution: {e}")

    def retrieve_relevant_memories(self, query_task: str, top_k: int = 3) -> list:
        """
        Retrieves the most relevant memories by performing a vector search
        followed by a graph traversal on the unified knowledge graph.
        """
        query_vector = self.model.encode(query_task)

        all_memory_nodes = self.graph_data_manager.query_nodes("node_type", "MemoryNote")

        if not all_memory_nodes:
            return []

        # 1. Find entry points via vector similarity
        nodes = [MemoryNote.from_node_attributes(node_id, self.graph_data_manager.get_node(node_id)) for node_id in all_memory_nodes]

        node_vectors = np.array([n.embedding for n in nodes if n.embedding.size > 0])
        if node_vectors.size == 0:
            return []

        similarities = self._cosine_similarity(query_vector, node_vectors)
        top_node_indices = np.argsort(similarities)[-top_k:][::-1]

        # 2. Perform graph traversal from entry points
        entry_point_ids = {nodes[i].id for i in top_node_indices}
        traversed_nodes_ids = set()

        for node_id in entry_point_ids:
            traversed_nodes_ids.add(node_id)
            neighbors = self.graph_data_manager.get_neighbors(node_id)
            for neighbor_id in neighbors:
                traversed_nodes_ids.add(neighbor_id)

        # 3. Format results
        amem_results = []
        for node_id in traversed_nodes_ids:
            node_data = self.graph_data_manager.get_node(node_id)
            if node_data:
                note = MemoryNote.from_node_attributes(node_id, node_data)
                result_str = (
                    f"Memory Note (ID: {note.id})\\n"
                    f"  Content: {note.content}\\n"
                    f"  Description: {note.contextual_description}\\n"
                    f"  Keywords: {', '.join(note.keywords)}"
                )
                amem_results.append(result_str)

        return amem_results

    def _cosine_similarity(self, vec_a, vec_b):
        """Computes cosine similarity between a vector and a matrix of vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b, axis=1)
        return np.dot(vec_b, vec_a) / (norm_a * norm_b)