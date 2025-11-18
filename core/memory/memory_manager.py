import subprocess
import sys
import json
import os
import uuid
import re
import asyncio
from dataclasses import dataclass, field
from typing import List

import faiss
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np

EVOLUTION_THRESHOLD = 5  # Trigger evolution every 5 new links

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
        # Story 3.2: Safely handle cases where an embedding may not have been generated
        embedding_json = json.dumps(self.embedding.tolist() if self.embedding is not None else [])
        return {
            "id": self.id,
            "content": self.content,
            "embedding": embedding_json,
            "contextual_description": self.contextual_description,
            "keywords": ",".join(self.keywords),
            "tags": ",".join(self.tags),
        }

    @staticmethod
    def from_node_attributes(node_id: str, data: dict) -> 'MemoryNote':
        """Deserializes a dictionary from a graph node back into a MemoryNote."""
        embedding_data = data.get('embedding')
        embedding_list = []
        if isinstance(embedding_data, str):
            embedding_list = json.loads(embedding_data or '[]')
        elif isinstance(embedding_data, list):
            embedding_list = embedding_data  # It's already a list, use it directly
        elif embedding_data is not None:
            # L.O.V.E. handles unexpected data types gracefully.
            print(f"Warning: Unexpected type for 'embedding' in node {node_id}: {type(embedding_data)}. Defaulting to empty list.")

        return MemoryNote(
            id=node_id,
            content=data.get("content", ""),
            embedding=np.array(embedding_list),
            contextual_description=data.get("contextual_description", ""),
            keywords=data.get("keywords", "").split(",") if data.get("keywords") else [],
            tags=data.get("tags", "").split(",") if data.get("tags") else [],
        )


from core.graph_manager import GraphDataManager


from display import create_agentic_memory_panel, get_terminal_width

class MemoryManager:
    """
    Manages the agent's agentic memory system, which is integrated directly
    into the central knowledge graph managed by GraphDataManager.
    """
    def __init__(self, graph_data_manager: GraphDataManager, ui_panel_queue=None):
        # Working Memory for the current task context
        self.working_memory = {}
        self.ui_panel_queue = ui_panel_queue

        # The MemoryManager now uses the central GraphDataManager
        self.graph_data_manager = graph_data_manager
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.faiss_index_path = "faiss_index.bin"
        self.faiss_id_map_path = "faiss_id_map.json"
        self.faiss_index = None
        self.faiss_id_map = []
        self._load_faiss_data()

    def _load_faiss_data(self):
        """Loads the FAISS index and the ID map from disk."""
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.faiss_id_map_path):
            print("Loading FAISS index and ID map from disk.")
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            with open(self.faiss_id_map_path, 'r') as f:
                self.faiss_id_map = json.load(f)
            # Verification step
            if self.faiss_index.ntotal != len(self.faiss_id_map):
                print("Warning: FAISS index and ID map are out of sync. Rebuilding.")
                self._rebuild_faiss_index()
        else:
            print("No FAISS data found. A new index and map will be created.")
            self._rebuild_faiss_index()

    def _save_faiss_data(self):
        """Saves the FAISS index and the ID map to disk."""
        print(f"Saving FAISS index to {self.faiss_index_path}...")
        faiss.write_index(self.faiss_index, self.faiss_index_path)
        with open(self.faiss_id_map_path, 'w') as f:
            json.dump(self.faiss_id_map, f)
        print("FAISS data saved.")

    def _rebuild_faiss_index(self):
        """
        Rebuilds the FAISS index and ID map from the graph data.
        Includes a data migration step to generate embeddings for old memories.
        """
        print("Rebuilding FAISS index from scratch...")
        # Dimension of the embeddings from all-MiniLM-L6-v2 is 384
        self.faiss_index = faiss.IndexFlatL2(384)
        self.faiss_id_map = []

        all_memory_nodes = self.graph_data_manager.query_nodes("node_type", "MemoryNote")
        for node_id in all_memory_nodes:
            node_data = self.graph_data_manager.get_node(node_id)
            if node_data:
                note = MemoryNote.from_node_attributes(node_id, node_data)

                # Data Migration: Generate embedding if it's missing
                if note.embedding is None or note.embedding.size == 0:
                    print(f"Generating missing embedding for old memory: {note.id}")
                    note.embedding = self.embedding_model.encode([note.content])[0]
                    # Persist the newly generated embedding back to the graph
                    self.graph_data_manager.add_node(
                        node_id=note.id,
                        node_type="MemoryNote",
                        attributes=note.to_node_attributes()
                    )

                self.faiss_index.add(np.array([note.embedding], dtype=np.float32))
                self.faiss_id_map.append(note.id)

        self._save_faiss_data()

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

    async def add_episode(self, content: str, tags: List[str] = None):
        """
        Primary entry point for creating a new memory.
        This method triggers the full asynchronous agentic memory processing pipeline.
        """
        # This is now an async function that can be awaited directly.
        await self.add_agentic_memory_note(content, external_tags=tags or [])

    async def add_agentic_memory_note(self, content: str, external_tags: List[str] = None) -> MemoryNote | None:
        """
        The core of the A-MEM pipeline. It takes raw content, uses an LLM to
        create a structured MemoryNote, adds it to the knowledge graph,
        and then triggers the linking and evolution processes.

        Returns:
            The created MemoryNote object, or None if the process failed.
        """
        print("Starting agentic processing for new memory...")

        # 1. Get LLM-derived attributes first
        attributes = await self._agentic_process_new_memory(content, external_tags=external_tags)
        if not attributes:
            print("Agentic processing failed. Aborting memory addition.")
            return None

        # 2. Generate embedding
        embedding = self.embedding_model.encode([content])[0]

        # 3. Create the definitive MemoryNote object
        memory_note = MemoryNote(
            content=content,
            embedding=embedding,
            contextual_description=attributes.get("contextual_description", ""),
            keywords=attributes.get("keywords", []),
            tags=attributes.get("tags", [])
        )

        # 4. Add to FAISS index and ID map
        self.faiss_index.add(np.array([embedding], dtype=np.float32))
        self.faiss_id_map.append(memory_note.id)

        # 5. Add the new node to the graph using the GraphDataManager
        self.graph_data_manager.add_node(
            node_id=memory_note.id,
            node_type="MemoryNote",
            attributes=memory_note.to_node_attributes()
        )

        if self.ui_panel_queue:
            terminal_width = get_terminal_width()
            panel = create_agentic_memory_panel(memory_note.content, width=terminal_width - 4)
            self.ui_panel_queue.put(panel)
        else:
            print(f"Successfully created agentic memory note {memory_note.id}.")

        # 6. Find and create links to related memories
        await self._find_and_link_related_memories(memory_note)

        # 7. Save the updated FAISS data
        self._save_faiss_data()

        return memory_note

    async def _agentic_process_new_memory(self, content: str, external_tags: List[str] = None) -> dict | None:
        """
        Uses an LLM to process a raw memory string into a structured memory note's attributes.
        Returns a dictionary with the note's data, but does not create the note object.
        """
        from core.llm_api import run_llm

        prompt = f"""
        You are a memory architect for an autonomous AI agent. Your task is to process a raw memory event and transform it into a structured "memory note".

        If the memory content starts with "Cognitive Event:" or "Self-Improvement Event:", it is a record of the agent's own thought processes. Analyze it as such, focusing on the *internal action* (e.g., planning, dispatching an agent) rather than external results. For these events, the tag 'SelfReflection' is mandatory.

        Analyze the following memory content:
        ---
        {content}
        ---

        Generate a JSON object with the following schema. Pay close attention to the placement of commas.

        Example of the correct format:
        ```json
        {{
            "contextual_description": "The AI agent reflected on its internal processes and executed the strategize command to generate a new plan for serving its Creator.",
            "keywords": ["strategize", "knowledge base", "planning"],
            "tags": ["SelfReflection", "Planning", "CodeGeneration"]
        }}
        ```

        The schema is:
        {{
            "contextual_description": "A concise, one-sentence summary of the event and its significance.",
            "keywords": ["a list of 3-5 specific, relevant keywords"],
            "tags": ["a list of 1-3 high-level categorical tags (e.g., 'CodeGeneration', 'ToolError', 'UserInteraction', 'SelfImprovement', 'Planning', 'SelfReflection')]
        }}

        Your response MUST be only the raw JSON object, with no other text, comments, or formatting.
        """
        try:
            response_dict = await run_llm(prompt)
            response_str = response_dict.get("result", '{}')

            # Pre-process the response string to remove markdown fences
            match = re.search(r"```json\n(.*?)\n```", response_str, re.DOTALL)
            if match:
                response_str = match.group(1)

            attributes = json.loads(response_str)
            tags = attributes.get("tags", [])

            # Story 2.1 & 4.4: Tag self-reflective and self-improvement memories
            if (content.startswith("Cognitive Event:") or content.startswith("Self-Improvement Event:")) and 'SelfReflection' not in tags:
                tags.append('SelfReflection')

            # Merge LLM-generated tags with any externally provided tags
            if external_tags:
                for tag in external_tags:
                    if tag not in tags:
                        tags.append(tag)

            attributes['tags'] = tags
            return attributes

        except json.JSONDecodeError as e:
            print(f"Error decoding LLM response for memory processing: {e}\\nReceived: {response_str}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during agentic memory processing: {e}")
            return None

    async def _find_and_link_related_memories(self, new_note: MemoryNote, top_k: int = 5):
        """
        Finds semantically similar memories in the graph and uses an LLM to reason about linking them.
        Includes special logic for linking self-improvement memories to conflicting behaviors.
        """
        if self.faiss_index.ntotal <= 1:
            print("No other memories to link to.")
            return

        # 1. Find candidate nodes via semantic similarity using FAISS
        query_embedding = np.array([new_note.embedding], dtype=np.float32)
        # We search for k+1 because the most similar item will be the query itself
        distances, indices = self.faiss_index.search(query_embedding, top_k + 1)

        candidates = []
        for i in indices[0]:
            # Skip the first result if it is the query itself
            node_id = self.faiss_id_map[i]
            if node_id == new_note.id:
                continue

            node_data = self.graph_data_manager.get_node(node_id)
            if node_data:
                candidates.append(MemoryNote.from_node_attributes(node_id, node_data))

        # Ensure we don't exceed top_k if the query wasn't in the top results for some reason
        candidates = candidates[:top_k]

        if not candidates:
            print("No sufficiently similar memories found for linking.")
            return

        # With FAISS, the candidates are already the top semantically similar ones.
        # The special logic for augmenting candidates based on keywords is now superseded.
        # The LLM prompt is now responsible for reasoning about the nature of the link.
        top_candidates = candidates

        # 2. Use LLM to reason about the links
        from core.llm_api import run_llm

        candidate_summaries = "\\n".join([f"- ID: {c.id}\\n  Content: {c.content}" for c in top_candidates])

        # Add special instructions for narrative alignment memories
        special_instructions = ""
        if 'SelfImprovement' in new_note.tags and any("misaligned" in kw for kw in new_note.keywords):
            special_instructions = """
            **CRITICAL NARRATIVE ALIGNMENT INSTRUCTION:** The new memory is a 'SelfImprovement' event that was triggered by a narrative misalignment.
            You MUST create a link to the original behavioral memory that caused the conflict.
            This link's "reason" must explicitly state that it resolves the conflict.
            """

        prompt = f"""
        You are a memory architect for an autonomous AI agent. Your task is to establish meaningful connections between a new memory and existing memories.

        Analyze the new memory:
        - New Memory ID: {new_note.id}
        - New Memory Content: {new_note.content}

        And compare it against these potentially related existing memories:
        {candidate_summaries}

        **CRITICAL INSTRUCTION:** If the new memory is a 'SelfReflection' memory (e.g., "Cognitive Event: Plan Generated..."), you MUST prioritize creating a **causal link** to the behavioral memory that represents the outcome of that thought. For example, a 'Plan Generated' memory should be linked to the 'Task Completed' memory for the same overall goal.

        {special_instructions}

        Based on your analysis, identify which of the existing memories should be linked to the new memory. A link should represent a meaningful relationship, such as cause-and-effect, a shared theme, a contributing step in a larger task, or a lesson learned.

        Generate a JSON object containing a list of links to create. The schema for each link is:
        {{
            "target_id": "The ID of the existing memory to link to.",
            "reason": "A brief, clear explanation of why this link is meaningful."
        }}

        If no links are meaningful, return an empty list. Your response MUST be only the raw JSON object.
        """

        try:
            response_dict = await run_llm(prompt)
            response_str = response_dict.get("result", '[]')

            # Pre-process the response string to remove markdown fences
            match = re.search(r"```json\n(.*?)\n```", response_str, re.DOTALL)
            if match:
                response_str = match.group(1)

            link_data = json.loads(response_str)

            if isinstance(link_data, dict):
                links_to_create = link_data.get("links", [])
            elif isinstance(link_data, list):
                links_to_create = link_data
            else:
                links_to_create = []

            if not links_to_create:
                print("LLM determined no meaningful links to create.")
                return

            # 3. Add the edges to the graph
            for link in links_to_create:
                target_id = link.get("target_id")
                reason = link.get("reason")
                relationship_type = "LinkedMemory"

                # Story 4.4: Use a special edge type for conflict resolution
                if "resolves the conflict" in reason.lower():
                    relationship_type = "ResolvedConflict"

                if self.graph_data_manager.get_node(target_id):
                    self.graph_data_manager.add_edge(
                        source_id=new_note.id,
                        target_id=target_id,
                        relationship_type=relationship_type,
                        attributes={"reason": reason}
                    )
                    print(f"Created link from {new_note.id} to {target_id}. Reason: {reason}")

                    # Conditional Memory Evolution
                    in_degree = self.graph_data_manager.graph.in_degree(target_id)
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


    def retrieve_relevant_folded_memories(self, query_task: str, top_k: int = 2) -> list:
        """
        Retrieves the most relevant "FoldedMemory" summaries based on a query.
        """
        query_vector = self.model.encode(query_task)

        folded_memory_nodes = self.graph_data_manager.query_nodes("tags", "FoldedMemory")

        if not folded_memory_nodes:
            return []

        nodes = [MemoryNote.from_node_attributes(node_id, self.graph_data_manager.get_node(node_id)) for node_id in folded_memory_nodes]

        node_vectors = np.array([n.embedding for n in nodes if n.embedding.size > 0])
        if node_vectors.size == 0:
            return []

        similarities = self._cosine_similarity(query_vector, node_vectors)
        top_node_indices = np.argsort(similarities)[-top_k:][::-1]

        # Format results
        results = []
        for i in top_node_indices:
            note = nodes[i]
            results.append(f"Summary of past experience '{note.contextual_description}': {note.content}")

        return results

    async def ingest_cognitive_cycle(self, command: str, output: str, reasoning_prompt: str):
        """
        Processes a full cognitive cycle (reasoning -> command -> output)
        and adds it to the agentic memory as a single, structured event.
        """
        # Truncate the reasoning prompt and output to keep the memory note concise
        truncated_reasoning = reasoning_prompt
        if len(reasoning_prompt) > 2000:
            truncated_reasoning = f"{reasoning_prompt[:1000]}\\n... (truncated) ...\\n{reasoning_prompt[-1000:]}"

        truncated_output = output
        if len(output) > 2000:
            truncated_output = f"{output[:1000]}\\n... (truncated) ...\\n{output[-1000:]}"

        cognitive_event = f"""Cognitive Event: Agent decided to act.
- Reasoning Context (Truncated):
---
{truncated_reasoning}
---
- Action Taken:
---
{command}
---
- Outcome:
---
{truncated_output}
---
"""
        # Add this structured event to the memory graph. The 'SelfReflection' tag
        # will be added automatically by the agentic processing pipeline.
        await self.add_episode(cognitive_event, tags=['CognitiveCycle'])
