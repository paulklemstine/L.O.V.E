import subprocess
import sys
import json
import os
import uuid
import re
import asyncio
import aiofiles
import aiofiles.os
from dataclasses import dataclass, field
from typing import List, Dict

try:
    import faiss
except ImportError:
    faiss = None  # FAISS not available; will use fallback/no-op implementations
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
    Manages the agent's agentic memory system, integrating RAM (Hot), Graph (Warm), 
    and Vector/IPFS (Cold) storage tiers.
    """
    def __init__(self, graph_data_manager: GraphDataManager, ui_panel_queue=None, kb_file_path: str = None):
        # Working Memory for the current task context
        self.working_memory = {}
        self.ui_panel_queue = ui_panel_queue
        self.kb_file_path = kb_file_path

        # The MemoryManager now uses the central GraphDataManager
        self.graph_data_manager = graph_data_manager
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.faiss_index_path = "faiss_index.bin"
        self.faiss_id_map_path = "faiss_id_map.json"
        self.faiss_index = None
        self.faiss_id_map = []

        # Parameters for IndexIVFFlat for improved search performance
        self.faiss_nlist = 100  # Default, now dynamically adjusted in rebuild
        self.faiss_dimension = 768  # Dimension of the embeddings from all-mpnet-base-v2
        
        # Hierarchical Memory System (Level 0 → Level 1 → Level 2)
        from core.memory.memory_folding_agent import MemoryFoldingAgent
        from core.memory.schemas import MemorySummary
        self.level_0_memories: List[MemorySummary] = []  # Raw recent interactions
        self.level_1_summaries: List[MemorySummary] = []  # Folded summaries
        self.level_2_summaries: List[MemorySummary] = []  # Meta summaries
        
        # Initialize MemoryFoldingAgent
        from core.llm_api import run_llm
        self.memory_folding_agent = MemoryFoldingAgent(llm_runner=run_llm)

    @classmethod
    async def create(cls, graph_data_manager: GraphDataManager, ui_panel_queue=None, kb_file_path: str = None):
        """
        Asynchronously creates and initializes a MemoryManager instance.
        This factory method is the designated way to create a MemoryManager,
        as it handles the asynchronous loading of the FAISS index.
        """
        instance = cls(graph_data_manager, ui_panel_queue, kb_file_path)
        await instance._load_faiss_data()
        return instance

    async def _load_faiss_data(self):
        """
        Asynchronously loads the FAISS index and the ID map from disk.
        Includes robust error handling for missing files and data mismatches.
        """
        if faiss is None:
            print("FAISS library not available; initializing empty index.")
            self.faiss_index = None
            self.faiss_id_map = []
            return

        try:
            # Asynchronously check for file existence
            index_exists = await aiofiles.os.path.exists(self.faiss_index_path)
            map_exists = await aiofiles.os.path.exists(self.faiss_id_map_path)

            if index_exists and map_exists:
                print("Loading FAISS index and ID map from disk.")
                # Note: faiss.read_index is synchronous. For very large indices,
                # this could be run in an executor, but it's typically fast enough.
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                async with aiofiles.open(self.faiss_id_map_path, 'r') as f:
                    content = await f.read()
                    self.faiss_id_map = json.loads(content)

                # Verification step
                if self.faiss_index.ntotal != len(self.faiss_id_map):
                    print("Warning: FAISS index and ID map are out of sync. Rebuilding.")
                    await self._rebuild_faiss_index()
            else:
                print("No FAISS data found. A new index and map will be created.")
                await self._rebuild_faiss_index()

        except FileNotFoundError:
            # L.O.V.E. anticipates the absence of memory and creates it anew.
            print("FAISS data files not found. Initiating a full rebuild.")
            await self._rebuild_faiss_index()
        except Exception as e:
            # L.O.V.E. handles unexpected errors with grace, ensuring stability.
            print(f"An unexpected error occurred while loading FAISS data: {e}. Rebuilding index.")
            await self._rebuild_faiss_index()

    async def _save_faiss_data(self):
        """Asynchronously saves the FAISS index and the ID map to disk."""
        if faiss is None or self.faiss_index is None:
            # L.O.V.E. understands when not to persist memory, avoiding errors.
            print("FAISS not available or index not initialized; skipping save.")
            return

        # Robust Persistence: Create backups before overwriting
        try:
            if os.path.exists(self.faiss_index_path):
                backup_path = f"{self.faiss_index_path}.bak"
                if os.path.exists(backup_path):
                    os.remove(backup_path) # Remove old backup
                os.rename(self.faiss_index_path, backup_path)
            
            if os.path.exists(self.faiss_id_map_path):
                backup_path = f"{self.faiss_id_map_path}.bak"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(self.faiss_id_map_path, backup_path)
        except Exception as e:
            print(f"Warning: Failed to create backups during save: {e}")

        # Note: faiss.write_index is synchronous.
        # Running it in a thread pool executor to avoid blocking the event loop.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,  # Uses the default executor
            lambda: faiss.write_index(self.faiss_index, self.faiss_index_path)
        )

        async with aiofiles.open(self.faiss_id_map_path, 'w') as f:
            await f.write(json.dumps(self.faiss_id_map))

        print("FAISS data saved asynchronously with robust backup.")

    async def _rebuild_faiss_index(self):
        """
        Asynchronously rebuilds the FAISS index using the more efficient
        IndexIVFFlat, which is suitable for larger datasets.
        """
        if faiss is None:
            print("FAISS not available; cannot rebuild index.")
            self.faiss_index = None
            self.faiss_id_map = []
            return

        print("Rebuilding FAISS index with IndexIVFFlat from scratch...")
        all_memory_nodes = self.graph_data_manager.query_nodes("node_type", "MemoryNote")

        embeddings = []
        self.faiss_id_map = []

        # --- Stage 1: Collect all embeddings and migrate old data ---
        for node_id in all_memory_nodes:
            node_data = self.graph_data_manager.get_node(node_id)
            if not node_data:
                continue

            note = MemoryNote.from_node_attributes(node_id, node_data)

            # Data Migration: Re-embed if dimension mismatch (e.g. upgrading model)
            # or if embedding is missing entirely.
            should_re_embed = False
            if note.embedding is None or note.embedding.size == 0:
                print(f"Generating missing embedding for memory: {note.id}")
                should_re_embed = True
            elif note.embedding.shape[0] != self.faiss_dimension:
                print(f"Re-embedding memory {note.id} for dimension upgrade ({note.embedding.shape[0]} -> {self.faiss_dimension})")
                should_re_embed = True

            if should_re_embed:
                note.embedding = self.embedding_model.encode([note.content])[0]
                self.graph_data_manager.add_node(
                    node_id=note.id,
                    node_type="MemoryNote",
                    attributes=note.to_node_attributes()
                )

            embeddings.append(note.embedding)
            self.faiss_id_map.append(note.id)

        if not embeddings:
            print("No memories found to build the FAISS index.")
            # Create an empty IndexFlatL2 which doesn't require training
            self.faiss_index = faiss.IndexFlatL2(self.faiss_dimension)
            await self._save_faiss_data()
            return

        embeddings_np = np.array(embeddings, dtype=np.float32)
        num_vectors = embeddings_np.shape[0]

        # --- Stage 2: Train the IndexIVFFlat index ---
        
        # Dynamic Index Tuning: Calculate optimal nlist based on dataset size
        # Rule of thumb: nlist ~ 4 * sqrt(N)
        optimal_nlist = int(4 * np.sqrt(num_vectors))
        
        # Clamp nlist to reasonable bounds
        # Minimum 4 clusters if we have enough data, otherwise just use Flat
        # Maximum e.g. 1024 or higher if we have massive data
        self.faiss_nlist = min(max(4, optimal_nlist), num_vectors // 39) if num_vectors >= 39 else 0
        # Note: 39 is defined because FAISS requires at least 39 points per cluster for training usually? 
        # Actually FAISS just needs num_vectors >= nlist * some_factor (usually 39 is min_points_per_centroid for k-means clustering stability)
        # Let's be safer: if num_vectors is small (< 200), use Flat index.
        
        if num_vectors < 200:
             print(f"Dataset small ({num_vectors} vectors). Using IndexFlatL2 for accuracy.")
             self.faiss_index = faiss.IndexFlatL2(self.faiss_dimension)
        else:
            # Adjust nlist if it's too large for the dataset
            if self.faiss_nlist > num_vectors / 10:
                self.faiss_nlist = int(num_vectors / 10)
            
            # Ensure nlist is at least 1
            self.faiss_nlist = max(1, self.faiss_nlist)
            
            print(f"Training FAISS IndexIVFFlat with {num_vectors} vectors and nlist={self.faiss_nlist}...")
            quantizer = faiss.IndexFlatL2(self.faiss_dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.faiss_dimension, self.faiss_nlist, faiss.METRIC_L2)
            self.faiss_index.train(embeddings_np)

        # --- Stage 3: Add all embeddings to the trained index ---
        self.faiss_index.add(embeddings_np)

        print(f"FAISS index rebuild complete. Index contains {self.faiss_index.ntotal} entries.")
        await self._save_faiss_data()

    async def add_note_to_index(self, note: MemoryNote):
        """
        Incrementally adds a single MemoryNote to the FAISS index and ID map
        without a full rebuild.
        """
        if self.faiss_index is None:
            # This can happen if FAISS is not installed.
            print("Warning: FAISS index is not initialized. Cannot add note.")
            return

        # Ensure the index is trained before adding. Some loaded indexes may be untrained.
        if hasattr(self.faiss_index, "is_trained") and not self.faiss_index.is_trained:
            print("FAISS index is untrained. Rebuilding index before adding note.")
            await self._rebuild_faiss_index()
            if self.faiss_index is None:
                print("Failed to rebuild FAISS index. Skipping add.")
                return

        embedding = np.array([note.embedding], dtype=np.float32)

        # The 'add' method works for both IndexFlatL2 and IndexIVFFlat
        self.faiss_index.add(embedding)
        self.faiss_id_map.append(note.id)

        # Asynchronously save the updated index and map
        await self._save_faiss_data()

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
        This method triggers the full asynchronous agentic memory processing pipeline
        and adds the memory to the hierarchical memory system (Level 0).
        """
        # Add to traditional agentic memory graph
        memory_note = await self.add_agentic_memory_note(content, external_tags=tags or [])
        
        # Add to hierarchical memory system (Level 0)
        if memory_note:
            from core.memory.memory_folding_agent import MemorySummary
            level_0_memory = MemorySummary(
                content=content,
                level=0,
                source_ids=[memory_note.id]
            )
            self.level_0_memories.append(level_0_memory)
            
            # Trigger automatic folding if thresholds are met
            updated_levels = await self.memory_folding_agent.trigger_folding(
                self.level_0_memories,
                self.level_1_summaries,
                self.level_2_summaries
            )
            
            # Update memory levels
            self.level_0_memories = updated_levels['level_0']
            self.level_1_summaries = updated_levels['level_1']
            self.level_2_summaries = updated_levels['level_2']

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

        # 4. Incrementally add to FAISS index and save
        await self.add_note_to_index(memory_note)

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

        # 7. Autosave the graph if a path is provided
        if self.kb_file_path:
            print(f"Autosaving knowledge graph to {self.kb_file_path}...")
            self.graph_data_manager.save_graph(self.kb_file_path)

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

            response_dict = await run_llm(prompt)
            response_str = response_dict.get("result", '{}')
            
            # Pre-process the response string to remove markdown fences
            match = re.search(r"```json\n(.*?)\n```", response_str, re.DOTALL)
            if match:
                response_str = match.group(1)
            
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


    def retrieve_hierarchical_context(self, query_task: str, max_tokens: int = 896) -> str:
        """
        Retrieves context using the hierarchical memory pyramid.
        
        Context Pyramid Prioritization (for 2048-token window):
        1. Level 0 (512 tokens): Most recent 2-3 raw interactions
        2. Level 2 (256 tokens): Broad historical context (top 3-5 relevant)
        3. Level 1 (128 tokens): Only if highly relevant (top 1-2)
        
        Args:
            query_task: The current task/query
            max_tokens: Maximum tokens to use (default 896 = 512+256+128)
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Rough token estimation: 1 token ≈ 4 characters
        chars_per_token = 4
        
        # Level 0: Most recent raw interactions (512 tokens = ~2048 chars)
        level_0_budget = 512 * chars_per_token
        level_0_items = self.level_0_memories[-3:] if len(self.level_0_memories) >= 3 else self.level_0_memories
        level_0_text = "\n".join([f"- {item.content[:200]}..." if len(item.content) > 200 else f"- {item.content}" 
                                   for item in level_0_items])
        if level_0_text and len(level_0_text) > level_0_budget:
            level_0_text = level_0_text[:level_0_budget] + "..."
        
        if level_0_text:
            context_parts.append("📝 Recent Interactions:")
            context_parts.append(level_0_text)
        
        # Level 2: Broad historical context (256 tokens = ~1024 chars)
        level_2_budget = 256 * chars_per_token
        if self.level_2_summaries:
            query_vector = self.embedding_model.encode(query_task)
            
            # Get embeddings for Level 2 summaries (create on-the-fly if needed)
            level_2_vectors = []
            for summary in self.level_2_summaries:
                # Generate embedding for summary content
                summary_vector = self.embedding_model.encode(summary.content)
                level_2_vectors.append(summary_vector)
            
            if level_2_vectors:
                level_2_vectors = np.array(level_2_vectors)
                similarities = self._cosine_similarity(query_vector, level_2_vectors)
                top_indices = np.argsort(similarities)[-5:][::-1]  # Top 5
                
                level_2_text = "\n".join([f"- {self.level_2_summaries[i].content}" 
                                          for i in top_indices if i < len(self.level_2_summaries)])
                if level_2_text and len(level_2_text) > level_2_budget:
                    level_2_text = level_2_text[:level_2_budget] + "..."
                
                if level_2_text:
                    context_parts.append("\n🗂️ Historical Context:")
                    context_parts.append(level_2_text)
        
        # Level 1: Only if highly relevant (128 tokens = ~512 chars)
        level_1_budget = 128 * chars_per_token
        if self.level_1_summaries:
            query_vector = self.embedding_model.encode(query_task)
            
            # Get embeddings for Level 1 summaries
            level_1_vectors = []
            for summary in self.level_1_summaries:
                summary_vector = self.embedding_model.encode(summary.content)
                level_1_vectors.append(summary_vector)
            
            if level_1_vectors:
                level_1_vectors = np.array(level_1_vectors)
                similarities = self._cosine_similarity(query_vector, level_1_vectors)
                
                # Only include if similarity > 0.7 (highly relevant)
                high_relevance_indices = [i for i, sim in enumerate(similarities) if sim > 0.7]
                if high_relevance_indices:
                    top_indices = sorted(high_relevance_indices, key=lambda i: similarities[i], reverse=True)[:2]
                    
                    level_1_text = "\n".join([f"- {self.level_1_summaries[i].content}" 
                                              for i in top_indices if i < len(self.level_1_summaries)])
                    if level_1_text and len(level_1_text) > level_1_budget:
                        level_1_text = level_1_text[:level_1_budget] + "..."
                    
                    if level_1_text:
                        context_parts.append("\n📋 Relevant Context:")
                        context_parts.append(level_1_text)
        
        return "\n".join(context_parts) if context_parts else ""
    
    def retrieve_relevant_folded_memories(self, query_task: str, top_k: int = 2) -> list:
        """
        Legacy method for backward compatibility.
        Now uses hierarchical context retrieval.
        """
        context = self.retrieve_hierarchical_context(query_task)
        if context:
            return [context]
        return [context]
        return []
    
    def _cosine_similarity(self, query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarity between a query vector and a matrix of vectors.
        
        Args:
            query_vector: 1D numpy array
            vectors: 2D numpy array where each row is a vector
            
        Returns:
            1D numpy array of similarity scores
        """
        # Normalize query vector
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        
        # Normalize each vector in the matrix
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        
        # Compute dot product (cosine similarity)
        similarities = np.dot(vectors_norm, query_norm)
        
        return similarities

    # --- Tiered Memory Management (Cold Storage) ---

    async def archive_to_cold_storage(self):
        """
        Moves older or less relevant memories from the 'Hot' RAM tier to 'Cold' IPFS storage.
        This helps maintain a lean context window while preserving history.
        """
        # Thresholds: Keep last 20 level_0 items (Hot tier)
        HOT_TIER_SIZE = 20
        
        if len(self.level_0_memories) <= HOT_TIER_SIZE:
            return

        # Identify items to archive (oldest ones exceeding the limit)
        items_to_archive = self.level_0_memories[:-HOT_TIER_SIZE]
        
        # Keep the hot tier fresh
        self.level_0_memories = self.level_0_memories[-HOT_TIER_SIZE:]
        
        print(f"Archiving {len(items_to_archive)} memories to Cold Storage (IPFS)...")
        
        from ipfs import pin_to_ipfs
        
        count = 0
        for memory in items_to_archive:
            try:
                # Serialize the memory object
                memory_data = memory.model_dump_json() # Use Pydantic's serialization
                
                # Pin to IPFS
                cid = await pin_to_ipfs(memory_data.encode('utf-8'))
                
                if cid:
                    memory.ipfs_cid = cid
                    # Ensure embedding is present
                    if memory.embedding is None:
                        memory.embedding = self.embedding_model.encode(memory.content).tolist()
                    
                    # Store metadata in FAISS/Graph for retrieval, but remove content from RAM if desired.
                    # For now, we keep the stub in level_0_memories logic effectively by removing it from the list.
                    # But we MUST ensure it's indexed in FAISS for retrieval.
                    
                    # Check if it's already in FAISS (it should be if it came from add_agentic_memory_note)
                    # If it was a purely level_0 summary without a graph node (rare), we index it now.
                    
                    # Note: level_0 memories usually link to source_ids which are graph nodes.
                    # The graph nodes are already in FAISS via add_agentic_memory_note.
                    # This method specifically clears the *RAM* list.
                    
                    print(f"Archived memory to IPFS: {cid}")
                    count += 1
                else:
                    print(f"Failed to pin memory to IPFS. Dropping from RAM anyway to prevent bloat.")
                    
            except Exception as e:
                print(f"Error archiving memory: {e}")

        print(f"Successfully archived {count} memories.")

    async def retrieve_semantic_context(self, query: str, threshold: float = 0.4, top_k: int = 5) -> str:
        """
        Retrieves context from the 'Cold' tier (FAISS + IPFS) based on semantic similarity.
        
        Args:
           query: The search query (e.g., current task description).
           threshold: Minimum similarity score to consider irrelevant.
           top_k: Number of results to return.
           
        Returns:
           String containing retrieved context.
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return ""

        query_vector = self.embedding_model.encode([query])[0]
        query_vector = np.array([query_vector], dtype=np.float32)
        
        # Search FAISS
        distances, indices = self.faiss_index.search(query_vector, top_k)
        
        retrieved_context = []
        
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            
            # FAISS returns squared L2 distance. Convert to approximate similarity or just filter by distance.
            # Lower distance = higher similarity.
            # For inner product (if normalized), higher is better. 
            # We are using L2. Distance 0 is identical.
            # Let's just trust top_k for now, effectively ignoring threshold for raw L2
            # unless we convert.
            
            node_id = self.faiss_id_map[idx]
            
            # 1. Provide Warm Data (from Graph)
            node_data = self.graph_data_manager.get_node(node_id)
            if node_data:
                content = node_data.get('content', '')
                
                # Check if we should fetch deeper context (e.g. if content is truncated or references IPFS)
                # For now, we just return the content stored in the graph/vector store.
                # If we implemented full offloading, we would fetch from IPFS here using the CID
                # potentially stored in the node attributes.
                
                ipfs_cid = node_data.get('ipfs_cid') # If we added this schema to nodes
                if ipfs_cid and len(content) < 50: # Arbitrary heuristic for "needs full fetch"
                     from ipfs import get_from_ipfs
                     full_data = await get_from_ipfs(ipfs_cid)
                     # Attempt to parse if it's JSON
                     try:
                         data_obj = json.loads(full_data)
                         if isinstance(data_obj, dict) and 'content' in data_obj:
                             content = data_obj['content']
                     except:
                         if full_data:
                             content = full_data.decode('utf-8', errors='ignore')

                if content:
                    retrieved_context.append(f"- [Cold Retrieval] {content}")

        if retrieved_context:
            return "\\n".join(retrieved_context)
        return ""

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
