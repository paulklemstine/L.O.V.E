import subprocess
import sys
import json
import os
import uuid
import re
import asyncio
import aiofiles
import aiofiles.os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# V2 Holographic Memory imports
try:
    from core.memory.fractal_schemas import (
        SalienceScore, GoldenMoment, SceneNode, ArcNode, EpochNode,
        FractalTreeRoot, EpisodicBuffer, StateAnchor
    )
    from core.memory.salience_scorer import SalienceScorer
    FRACTAL_MEMORY_AVAILABLE = True
except ImportError:
    FRACTAL_MEMORY_AVAILABLE = False

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
        from core.memory.schemas import MemorySummary, WisdomEntry
        self.level_0_memories: List[MemorySummary] = []  # Raw recent interactions
        self.level_1_summaries: List[MemorySummary] = []  # Folded summaries
        self.level_2_summaries: List[MemorySummary] = []  # Meta summaries
        
        # Story 2.1: Wisdom Store for Recursive Learning
        # Stores distilled lessons learned from operational experience
        self.wisdom_store: List[WisdomEntry] = []
        self.wisdom_file_path = "wisdom_store.json"
        self._load_wisdom_store()  # Load persisted wisdom on init
        
        # Initialize MemoryFoldingAgent
        from core.llm_api import run_llm
        self.memory_folding_agent = MemoryFoldingAgent(llm_runner=run_llm)
        
        # V2 Holographic Memory: Fractal Tree Integration
        self._init_fractal_memory()

    async def save_activity_log(self, action: str, result: str, filepath=".ralph/activity_log.md"):
        """Appends a structured log entry to the activity log."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"## [{timestamp}] {action}\n**Result:** {result}\n\n"
        
        log_dir = os.path.dirname(filepath)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        async with aiofiles.open(filepath, 'a', encoding='utf-8') as f:
            await f.write(entry)
            
    async def read_recent_activity(self, filepath=".ralph/activity_log.md", n=5) -> str:
        """Reads the last N entries from the activity log."""
        if not os.path.exists(filepath):
            return ""
            
        async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
            content = await f.read()
            
        # Parse entries by "## ["
        entries = re.split(r'(?=## \[\d{4}-\d{2}-\d{2})', content)
        # Filter empty
        entries = [e for e in entries if e.strip()]
        
        recent = entries[-n:]
        return "".join(recent)

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
        # MEMORY LOGGING: Track all add_episode calls
        import traceback
        caller_info = ''.join(traceback.format_stack()[-4:-1])
        print(f"\n[MEMORY TRACE] add_episode called")
        print(f"  Tags: {tags}")
        print(f"  Content preview: {content[:100]}..." if len(content) > 100 else f"  Content: {content}")
        print(f"  Call stack (last 3 frames):\n{caller_info}")
        
        # Add to traditional agentic memory graph
        memory_note = await self.add_agentic_memory_note(content, external_tags=tags or [])
        
        # Add to hierarchical memory system (Level 0)
        if memory_note:
            print(f"[MEMORY TRACE] Memory note created successfully: {memory_note.id}")
            from core.memory.memory_folding_agent import MemorySummary
            level_0_memory = MemorySummary(
                content=content,
                level=0,
                source_ids=[memory_note.id]
            )
            self.level_0_memories.append(level_0_memory)
            print(f"[MEMORY TRACE] Level 0 memories count: {len(self.level_0_memories)}")
            
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
        else:
            print(f"[MEMORY TRACE] WARNING: add_agentic_memory_note returned None!")

    async def add_agentic_memory_note(self, content: str, external_tags: List[str] = None) -> MemoryNote | None:
        """
        The core of the A-MEM pipeline. It takes raw content, uses an LLM to
        create a structured MemoryNote, adds it to the knowledge graph,
        and then triggers the linking and evolution processes.

        Returns:
            The created MemoryNote object, or None if the process failed.
        """
        print(f"[MEMORY] Starting agentic processing for new memory...")
        print(f"[MEMORY]   External tags: {external_tags}")
        print(f"[MEMORY]   Content length: {len(content)} chars")

        # 1. Get LLM-derived attributes first
        attributes = await self._agentic_process_new_memory(content, external_tags=external_tags)
        if not attributes:
            print("[MEMORY] ERROR: Agentic processing failed (LLM returned invalid attributes). Aborting memory addition.")
            return None
        print(f"[MEMORY]   LLM attributes: contextual_description='{attributes.get('contextual_description', 'N/A')[:50]}...'")

        # 2. Generate embedding
        embedding = self.embedding_model.encode([content])[0]
        print(f"[MEMORY]   Embedding generated: shape={embedding.shape}")

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
        
        print(f"[MEMORY] SUCCESS: Created agentic memory note {memory_note.id}")
        print(f"[MEMORY]   FAISS index now has {self.faiss_index.ntotal if self.faiss_index else 0} entries")
        print(f"[MEMORY]   ID map now has {len(self.faiss_id_map)} entries")

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

    # =========================================================================
    # Story 2.1: Wisdom Store - Recursive Learning
    # =========================================================================

    def _load_wisdom_store(self) -> None:
        """Loads persisted wisdom entries from disk."""
        from core.memory.schemas import WisdomEntry
        
        if os.path.exists(self.wisdom_file_path):
            try:
                with open(self.wisdom_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.wisdom_store = [WisdomEntry(**entry) for entry in data]
                print(f"📚 Loaded {len(self.wisdom_store)} wisdom entries from {self.wisdom_file_path}")
            except Exception as e:
                print(f"Warning: Could not load wisdom store: {e}")
                self.wisdom_store = []
        else:
            self.wisdom_store = []
    
    def _save_wisdom_store(self) -> None:
        """Persists wisdom entries to disk."""
        try:
            data = [entry.model_dump() for entry in self.wisdom_store]
            with open(self.wisdom_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Saved {len(self.wisdom_store)} wisdom entries to {self.wisdom_file_path}")
        except Exception as e:
            print(f"Warning: Could not save wisdom store: {e}")
    
    def add_wisdom(self, wisdom: 'WisdomEntry') -> None:
        """
        Adds a new wisdom entry to the store.
        
        Story 2.1: Wisdom entries are distilled lessons learned that update
        the prompt context dynamically for recursive improvement.
        
        Args:
            wisdom: A WisdomEntry object containing the distilled lesson.
        """
        from core.memory.schemas import WisdomEntry
        
        # Generate embedding for semantic retrieval
        if wisdom.embedding is None:
            wisdom.embedding = self.embedding_model.encode([wisdom.principle]).tolist()[0]
        
        self.wisdom_store.append(wisdom)
        
        # Keep top 50 entries by confidence, removing lowest if over limit
        if len(self.wisdom_store) > 50:
            self.wisdom_store = sorted(
                self.wisdom_store,
                key=lambda w: w.confidence,
                reverse=True
            )[:50]
        
        # Persist to disk
        self._save_wisdom_store()
        print(f"🧠 Added wisdom: '{wisdom.principle[:50]}...' (confidence: {wisdom.confidence})")
    
    def get_relevant_wisdom(
        self, 
        context: str, 
        top_k: int = 5
    ) -> List['WisdomEntry']:
        """
        Retrieves the top K most relevant wisdom entries for the given context.
        
        Story 2.1: Uses semantic similarity to find wisdom entries that are
        most applicable to the current situation.
        
        Args:
            context: The current task/situation description
            top_k: Number of wisdom entries to retrieve (default 5)
            
        Returns:
            List of WisdomEntry objects, sorted by relevance.
        """
        from core.memory.schemas import WisdomEntry
        
        if not self.wisdom_store:
            return []
        
        # Generate embedding for context
        context_embedding = self.embedding_model.encode([context])[0]
        
        # Gather wisdom embeddings
        wisdom_with_embeddings = []
        for wisdom in self.wisdom_store:
            if wisdom.embedding is not None:
                wisdom_with_embeddings.append(wisdom)
            else:
                # Generate embedding on the fly
                wisdom.embedding = self.embedding_model.encode([wisdom.principle]).tolist()[0]
                wisdom_with_embeddings.append(wisdom)
        
        if not wisdom_with_embeddings:
            return []
        
        # Calculate similarities
        wisdom_embeddings = np.array([w.embedding for w in wisdom_with_embeddings])
        similarities = self._cosine_similarity(context_embedding, wisdom_embeddings)
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filter by minimum similarity threshold (0.3)
        relevant_wisdom = []
        for idx in top_indices:
            if similarities[idx] > 0.3:
                relevant_wisdom.append(wisdom_with_embeddings[idx])
        
        return relevant_wisdom
    
    def format_wisdom_for_prompt(self, wisdom_entries: List['WisdomEntry']) -> str:
        """
        Formats wisdom entries for injection into System Prompt.
        
        Story 2.1: Creates the "## Recursive Wisdom" section that enables
        the agent to reference lessons learned in its reasoning.
        
        Args:
            wisdom_entries: List of relevant WisdomEntry objects
            
        Returns:
            Formatted string ready for prompt injection.
        """
        if not wisdom_entries:
            return ""
        
        lines = ["## 🔄 Recursive Wisdom\n"]
        lines.append("The following lessons were learned from previous operational cycles:\n")
        
        for i, wisdom in enumerate(wisdom_entries, 1):
            lines.append(f"### Lesson {i} (Confidence: {wisdom.confidence:.0%})")
            lines.append(wisdom.to_prompt_format())
            lines.append("")  # Blank line separator
        
        lines.append("**Apply these principles to avoid repeating mistakes and reinforce successful patterns.**\n")
        
        return "\n".join(lines)
    
    async def extract_wisdom_from_episode(
        self, 
        episode_content: str,
        outcome: str = "unknown"
    ) -> 'WisdomEntry | None':
        """
        Uses an LLM to extract wisdom from an operational episode.
        
        Story 2.1: This is the core of the Ouroboros Memory Fold - converting
        raw experience into distilled wisdom that informs future decisions.
        
        Args:
            episode_content: Description of what happened
            outcome: Whether the episode was successful, failed, or unknown
            
        Returns:
            A WisdomEntry object if extraction was successful, None otherwise.
        """
        from core.llm_api import run_llm
        from core.memory.schemas import WisdomEntry
        
        prompt = f"""
You are the Wisdom Extractor for an autonomous AI agent. Your task is to distill a lesson learned from an operational episode.

## Episode Content
{episode_content}

## Outcome
{outcome}

## Task
Analyze this episode and extract a reusable principle that the agent should remember. The principle should be:
1. **Actionable**: Something the agent can apply in future situations
2. **Specific**: Not vague generalizations
3. **Derived from evidence**: Based on what actually happened

## Output Format
Return a JSON object with this schema:
{{
    "situation": "Brief description of the context/problem (1-2 sentences)",
    "action": "What action was taken (1 sentence)",
    "outcome": "What the result was (1 sentence)",
    "principle": "The lesson to remember and apply in future (1 sentence, imperative voice)",
    "confidence": 0.0-1.0,
    "source": "success" | "failure" | "experience",
    "tags": ["relevant", "categorization", "tags"]
}}

Return ONLY the JSON object.
"""
        
        try:
            response_dict = await run_llm(prompt)
            response_str = response_dict.get("result", '{}')
            
            # Clean markdown fences
            match = re.search(r"```json\n(.*?)\n```", response_str, re.DOTALL)
            if match:
                response_str = match.group(1)
            
            wisdom_data = json.loads(response_str)
            
            # Create WisdomEntry
            wisdom = WisdomEntry(
                situation=wisdom_data.get("situation", ""),
                action=wisdom_data.get("action", ""),
                outcome=wisdom_data.get("outcome", ""),
                principle=wisdom_data.get("principle", ""),
                confidence=float(wisdom_data.get("confidence", 0.7)),
                source=wisdom_data.get("source", "experience"),
                tags=wisdom_data.get("tags", [])
            )
            
            return wisdom
            
        except json.JSONDecodeError as e:
            print(f"Error parsing wisdom extraction response: {e}")
            return None
        except Exception as e:
            print(f"Error extracting wisdom: {e}")
            return None


    # --- Story 2.1: Semantic Memory Bridge ---

    async def search_similar_interactions(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Performs semantic search on the FAISS index to find similar past interactions.
        
        This is the core of the Semantic Memory Bridge (Story 2.1), enabling the
        Reasoning Agent to query past user stories or code patches before starting
        a task to avoid repeating mistakes or duplicating effort.
        
        Args:
            query: The user request or task description
            top_k: Number of similar results to return (default 3)
            
        Returns:
            List of dictionaries containing:
            - id: Memory note ID
            - content: The memory content (truncated for context efficiency)
            - similarity_score: Distance score from FAISS (lower is more similar)
            - keywords: Associated keywords
            - tags: Associated tags
            - contextual_description: LLM-generated description
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode([query])[0]
            query_embedding = np.array([query_embedding], dtype=np.float32)
            
            # Search FAISS index
            # Limit top_k to available items
            actual_k = min(top_k, self.faiss_index.ntotal)
            distances, indices = self.faiss_index.search(query_embedding, actual_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.faiss_id_map):
                    continue
                    
                node_id = self.faiss_id_map[idx]
                node_data = self.graph_data_manager.get_node(node_id)
                
                if node_data:
                    note = MemoryNote.from_node_attributes(node_id, node_data)
                    
                    # Truncate content for context efficiency (max 500 chars)
                    truncated_content = note.content[:500] + "..." if len(note.content) > 500 else note.content
                    
                    results.append({
                        "id": node_id,
                        "content": truncated_content,
                        "similarity_score": float(distances[0][i]),  # Lower = more similar
                        "keywords": note.keywords,
                        "tags": note.tags,
                        "contextual_description": note.contextual_description
                    })
            
            return results
            
        except Exception as e:
            print(f"Error in search_similar_interactions: {e}")
            return []

    def format_memory_context_for_prompt(self, similar_interactions: List[Dict]) -> str:
        """
        Formats similar interactions into a context string for injection into prompts.
        
        Args:
            similar_interactions: Results from search_similar_interactions()
            
        Returns:
            Formatted string suitable for prompt injection
        """
        if not similar_interactions:
            return ""
        
        context_parts = ["## 🧠 Relevant Past Interactions\n"]
        context_parts.append("The following past interactions may be relevant to this task:\n")
        
        for i, interaction in enumerate(similar_interactions, 1):
            context_parts.append(f"### Memory {i}")
            if interaction.get("contextual_description"):
                context_parts.append(f"**Summary:** {interaction['contextual_description']}")
            context_parts.append(f"**Content:** {interaction['content']}")
            if interaction.get("keywords"):
                context_parts.append(f"**Keywords:** {', '.join(interaction['keywords'])}")
            if interaction.get("tags"):
                context_parts.append(f"**Tags:** {', '.join(interaction['tags'])}")
            context_parts.append("")  # Blank line between entries
        
        return "\n".join(context_parts)

    # --- Story 2.2: Memory Folding Strategy ---

    def estimate_token_count(self, messages: List) -> int:
        """
        Estimates the token count for a list of messages.
        
        Uses a simple heuristic: ~4 characters per token (conservative estimate).
        This is faster than using a real tokenizer and accurate enough for thresholds.
        
        Args:
            messages: List of BaseMessage objects or strings
            
        Returns:
            Estimated token count
        """
        total_chars = 0
        
        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, str):
                content = msg
            elif isinstance(msg, dict):
                content = str(msg.get('content', msg))
            else:
                content = str(msg)
            
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, (list, dict)):
                total_chars += len(str(content))
        
        # Conservative estimate: 4 chars per token
        return total_chars // 4

    async def check_and_fold_context(
        self, 
        messages: List,
        token_limit: int = 4096,
        threshold: float = 0.8
    ) -> tuple:
        """
        Checks if context exceeds threshold and triggers folding.
        
        Story 2.2: When token count > 80% of limit:
        1. Triggers MemoryFoldingAgent to create a summary
        2. Replaces oldest 50% of messages with the summary
        3. Stores the Knowledge Nugget in the knowledge base
        
        Args:
            messages: List of BaseMessage objects
            token_limit: Maximum token limit (default 4096)
            threshold: Threshold percentage to trigger folding (default 0.8 = 80%)
            
        Returns:
            Tuple of (modified_messages, knowledge_nugget or None)
        """
        from core.memory.schemas import KnowledgeNugget
        from langchain_core.messages import SystemMessage
        
        current_tokens = self.estimate_token_count(messages)
        threshold_tokens = int(token_limit * threshold)
        
        if current_tokens <= threshold_tokens:
            # No folding needed
            return messages, None
        
        print(f"Memory folding triggered: {current_tokens} tokens > {threshold_tokens} threshold")
        
        # Determine how many messages to fold (oldest 50%)
        fold_count = len(messages) // 2
        if fold_count < 2:
            # Too few messages to fold meaningfully
            return messages, None
        
        messages_to_fold = messages[:fold_count]
        messages_to_keep = messages[fold_count:]
        
        # Create a summary of the folded messages
        try:
            summary = await self._create_fold_summary(messages_to_fold)
            
            if not summary:
                print("Warning: Failed to create fold summary, keeping original messages")
                return messages, None
            
            # Calculate token savings
            original_tokens = self.estimate_token_count(messages_to_fold)
            summary_tokens = self.estimate_token_count([summary])
            token_savings = original_tokens - summary_tokens
            
            # Create the Knowledge Nugget
            nugget = KnowledgeNugget(
                content=summary,
                source_message_count=fold_count,
                key_directives=self._extract_key_directives(messages_to_fold),
                topics=self._extract_topics(messages_to_fold),
                token_savings=token_savings
            )
            
            # Store nugget in knowledge base
            await self._store_knowledge_nugget(nugget)
            
            # Create a system message with the summary
            summary_message = SystemMessage(
                content=f"[Memory Folded: {fold_count} previous messages summarized]\n\n{summary}"
            )
            
            # Return modified messages: summary + kept messages
            modified_messages = [summary_message] + messages_to_keep
            
            print(f"Memory folding complete: Replaced {fold_count} messages with summary, saved ~{token_savings} tokens")
            
            return modified_messages, nugget
            
        except Exception as e:
            print(f"Error during memory folding: {e}")
            return messages, None

    async def _create_fold_summary(self, messages: List) -> str:
        """
        Creates a compressed summary of messages using LLM.
        
        Args:
            messages: Messages to summarize
            
        Returns:
            Summary string
        """
        from core.llm_api import run_llm
        
        # Format messages for the prompt
        formatted = []
        for msg in messages:
            role = "User" if hasattr(msg, 'type') and msg.type == 'human' else "Assistant"
            if hasattr(msg, 'content'):
                formatted.append(f"{role}: {msg.content[:500]}...")
        
        messages_text = "\n".join(formatted)
        
        prompt = f"""
        Summarize the following conversation thread into a concise "Knowledge Nugget".
        
        CRITICAL: Preserve any direct instructions, mandates, or goals from the user.
        Focus on: key decisions made, important context, and any ongoing tasks.
        
        Conversation:
        ---
        {messages_text}
        ---
        
        Provide a 2-3 sentence summary that captures the essential context.
        Do not include any preamble, just the summary.
        """
        
        try:
            response = await run_llm(prompt, purpose="memory_folding")
            return response.get("result", "").strip()
        except Exception as e:
            print(f"Error creating fold summary: {e}")
            return ""

    def _extract_key_directives(self, messages: List) -> List[str]:
        """
        Extracts key directives/instructions from messages.
        
        Looks for patterns indicating direct user instructions.
        """
        directives = []
        directive_keywords = ["must", "always", "never", "important", "critical", "priority"]
        
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if isinstance(content, str):
                    # Look for sentences with directive keywords
                    sentences = content.split('.')
                    for sentence in sentences:
                        sentence_lower = sentence.lower()
                        if any(kw in sentence_lower for kw in directive_keywords):
                            directive = sentence.strip()
                            if directive and len(directive) < 200:
                                directives.append(directive)
        
        # Limit to top 5 directives
        return directives[:5]

    def _extract_topics(self, messages: List) -> List[str]:
        """
        Extracts main topics from messages for tagging.
        """
        # Simple keyword extraction - in production, could use NLP
        topics = set()
        common_topics = [
            "code", "bug", "fix", "feature", "test", "deploy", 
            "memory", "agent", "tool", "api", "database", "ui",
            "bluesky", "post", "image", "evolution"
        ]
        
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            if isinstance(content, str):
                content_lower = content.lower()
                for topic in common_topics:
                    if topic in content_lower:
                        topics.add(topic)
        
        return list(topics)[:5]

    async def _store_knowledge_nugget(self, nugget) -> None:
        """
        Stores a Knowledge Nugget in the knowledge base and memory graph.
        """
        from core.memory.schemas import KnowledgeNugget
        
        try:
            # Store as a memory note with special tag
            await self.add_episode(
                content=f"[Knowledge Nugget] {nugget.content}\n\nKey Directives: {nugget.key_directives}\nTopics: {nugget.topics}",
                tags=["KnowledgeNugget", "FoldedMemory"] + nugget.topics
            )
            print(f"Stored Knowledge Nugget covering {nugget.source_message_count} messages")
        except Exception as e:
            print(f"Warning: Failed to store Knowledge Nugget: {e}")

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

    # =========================================================================
    # V2 HOLOGRAPHIC MEMORY: Fractal Tree Methods
    # =========================================================================
    
    def _init_fractal_memory(self):
        """
        Initialize V2 Holographic Memory components:
        - Fractal Tree (long-term hierarchical storage)
        - Episodic Buffer (medium-term working buffer)
        - State Anchor (identity injection)
        - Salience Scorer (preservation decisions)
        """
        if not FRACTAL_MEMORY_AVAILABLE:
            print("[FRACTAL MEMORY] Fractal memory schemas not available. Skipping initialization.")
            self.fractal_tree = None
            self.episodic_buffer = None
            self.state_anchor = None
            self.salience_scorer = None
            self.golden_moments = []
            return
        
        # File paths for persistence
        self.fractal_tree_path = "_memory_/fractal_tree.json"
        self.state_anchor_path = "_memory_/STATE_ANCHOR.md"
        self.episodic_buffer_path = "core/memory/episodic_buffer.json"
        
        # Initialize components
        self.salience_scorer = SalienceScorer()
        self.golden_moments: List[GoldenMoment] = []
        
        # Load persisted data
        self._load_fractal_tree()
        self._load_episodic_buffer()
        self._load_state_anchor()
        
        print(f"[FRACTAL MEMORY] Initialized with {len(self.golden_moments)} Golden Moments")
    
    def _load_fractal_tree(self):
        """Load fractal tree from disk."""
        if not FRACTAL_MEMORY_AVAILABLE:
            return
        
        try:
            if os.path.exists(self.fractal_tree_path):
                with open(self.fractal_tree_path, 'r') as f:
                    data = json.load(f)
                    self.fractal_tree = FractalTreeRoot(**data) if data.get('summary') else FractalTreeRoot()
                    # Load pinned crystals as Golden Moments
                    if data.get('pinned_crystals'):
                        for crystal_data in data['pinned_crystals']:
                            try:
                                self.golden_moments.append(GoldenMoment(**crystal_data))
                            except:
                                pass
            else:
                self.fractal_tree = FractalTreeRoot()
        except Exception as e:
            print(f"[FRACTAL MEMORY] Error loading fractal tree: {e}")
            self.fractal_tree = FractalTreeRoot()
    
    def _save_fractal_tree(self):
        """Persist fractal tree to disk."""
        if not FRACTAL_MEMORY_AVAILABLE or self.fractal_tree is None:
            return
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.fractal_tree_path), exist_ok=True)
            
            # Convert to dict with pinned crystals
            data = self.fractal_tree.model_dump()
            data['pinned_crystals'] = [gm.model_dump() for gm in self.golden_moments[:10]]  # Top 10
            
            with open(self.fractal_tree_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[FRACTAL MEMORY] Error saving fractal tree: {e}")
    
    def _load_episodic_buffer(self):
        """Load episodic buffer from disk."""
        if not FRACTAL_MEMORY_AVAILABLE:
            return
        
        try:
            if os.path.exists(self.episodic_buffer_path):
                with open(self.episodic_buffer_path, 'r') as f:
                    data = json.load(f)
                    self.episodic_buffer = EpisodicBuffer(**data)
            else:
                self.episodic_buffer = EpisodicBuffer()
        except Exception as e:
            print(f"[FRACTAL MEMORY] Error loading episodic buffer: {e}")
            self.episodic_buffer = EpisodicBuffer()
    
    def _save_episodic_buffer(self):
        """Persist episodic buffer to disk."""
        if not FRACTAL_MEMORY_AVAILABLE or self.episodic_buffer is None:
            return
        
        try:
            with open(self.episodic_buffer_path, 'w') as f:
                json.dump(self.episodic_buffer.model_dump(), f, indent=2)
        except Exception as e:
            print(f"[FRACTAL MEMORY] Error saving episodic buffer: {e}")
    
    def _load_state_anchor(self):
        """Load state anchor from disk."""
        if not FRACTAL_MEMORY_AVAILABLE:
            return
        
        try:
            self.state_anchor = StateAnchor()
            
            if os.path.exists(self.state_anchor_path):
                with open(self.state_anchor_path, 'r') as f:
                    content = f.read()
                    # Parse markdown to extract sections (simplified)
                    # For now, use defaults but preserve golden crystals
                    self.state_anchor.golden_crystals = self.golden_moments[:5]
            else:
                self._save_state_anchor()
        except Exception as e:
            print(f"[FRACTAL MEMORY] Error loading state anchor: {e}")
            self.state_anchor = StateAnchor()
    
    def _save_state_anchor(self):
        """Persist state anchor to disk."""
        if not FRACTAL_MEMORY_AVAILABLE or self.state_anchor is None:
            return
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.state_anchor_path), exist_ok=True)
            
            with open(self.state_anchor_path, 'w') as f:
                f.write(self.state_anchor.to_prompt_format())
        except Exception as e:
            print(f"[FRACTAL MEMORY] Error saving state anchor: {e}")
    
    def get_state_anchor_for_prompt(self) -> str:
        """
        Get the state anchor formatted for system prompt injection.
        
        This is called at the start of every interaction to maintain
        identity continuity across sessions.
        """
        if not FRACTAL_MEMORY_AVAILABLE or self.state_anchor is None:
            return ""
        
        # Update golden crystals before returning
        self.state_anchor.golden_crystals = self.golden_moments[:5]
        return self.state_anchor.to_prompt_format()
    
    async def add_to_episodic_buffer(self, content: str, metadata: Dict = None) -> bool:
        """
        Add content to the episodic buffer.
        
        Returns True if buffer is full and should be flushed to Arc.
        """
        if not FRACTAL_MEMORY_AVAILABLE or self.episodic_buffer is None:
            return False
        
        should_flush = self.episodic_buffer.add_episode(content, metadata)
        self._save_episodic_buffer()
        
        if should_flush:
            print(f"[FRACTAL MEMORY] Episodic buffer full ({len(self.episodic_buffer.buffer)} items). Triggering Arc creation.")
            await self.flush_episodic_buffer_to_arc()
            return True
        
        return False
    
    async def flush_episodic_buffer_to_scene(self) -> Optional[SceneNode]:
        """
        Flush the episodic buffer to create a new SceneNode.
        
        Story M.2: When episodic_buffer > 50 items:
        1. Score each episode for salience
        2. Extract high-salience items as crystals (never compressed)
        3. Summarize the rest into the scene
        """
        if not FRACTAL_MEMORY_AVAILABLE or self.episodic_buffer is None:
            return None
        
        if len(self.episodic_buffer.buffer) == 0:
            return None
        
        # Get episodes and clear buffer
        episodes = self.episodic_buffer.flush()
        self._save_episodic_buffer()
        
        # Use memory folding agent to create scene with salience scoring
        scene = await self.memory_folding_agent.fold_to_scene(episodes)
        
        if scene:
            # Store golden moments from scene crystals
            for crystal in scene.crystals:
                self.add_golden_moment(crystal)
            
            # TODO: Add scene to arc in fractal tree structure
            # For now, we just preserve the crystals
            self._save_fractal_tree()
            
            print(f"[FRACTAL MEMORY] Created Scene with {len(scene.crystals)} Golden Moments preserved")
        
        return scene
    
    def add_golden_moment(self, moment: GoldenMoment):
        """
        Add a Golden Moment to the permanent preservation store.
        These are NEVER compressed and survive all folding operations.
        """
        if not FRACTAL_MEMORY_AVAILABLE:
            return
        
        # Check for duplicates (by content similarity)
        for existing in self.golden_moments:
            if existing.raw_text == moment.raw_text:
                return  # Already preserved
        
        self.golden_moments.append(moment)
        self.fractal_tree.total_golden_moments = len(self.golden_moments)
        
        # Update state anchor with new crystals
        if self.state_anchor:
            self.state_anchor.golden_crystals = self.golden_moments[:5]
        
        self._save_fractal_tree()
        self._save_state_anchor()
        
        print(f"[GOLDEN MOMENT] Preserved: '{moment.raw_text[:50]}...' (Total: {len(self.golden_moments)})")
    
    async def score_and_preserve_if_golden(self, content: str, source_id: str = "") -> SalienceScore:
        """
        Score content for salience and preserve as Golden Moment if threshold met.
        
        This is the core of the Golden Moment Preservation Protocol.
        """
        if not FRACTAL_MEMORY_AVAILABLE or self.salience_scorer is None:
            # Return neutral score if fractal memory not available
            return SalienceScore() if FRACTAL_MEMORY_AVAILABLE else None
        
        score, golden = await self.salience_scorer.score_and_preserve(
            content, source_id, threshold=0.8
        )
        
        if golden:
            self.add_golden_moment(golden)
        
        return score
    
    def retrieve_golden_moments(self, query: str = "", top_k: int = 5) -> List[GoldenMoment]:
        """
        Retrieve relevant Golden Moments based on query.
        
        If no query provided, returns most recent.
        """
        if not FRACTAL_MEMORY_AVAILABLE or not self.golden_moments:
            return []
        
        if not query:
            return self.golden_moments[:top_k]
        
        # Simple keyword matching for now
        # TODO: Add semantic similarity search
        query_lower = query.lower()
        scored = []
        for gm in self.golden_moments:
            score = sum(1 for word in query_lower.split() if word in gm.raw_text.lower())
            if score > 0:
                scored.append((score, gm))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [gm for _, gm in scored[:top_k]]
    
    def get_fractal_context_for_query(self, query: str, max_chars: int = 2000) -> str:
        """
        Retrieve context from the fractal memory hierarchy for a query.
        
        Story M.3: Associative Diver Retrieval
        1. Check STATE_ANCHOR (always included)
        2. Check relevant Golden Moments
        3. TODO: Query the Fractal Tree with drill-down logic
        """
        if not FRACTAL_MEMORY_AVAILABLE:
            return ""
        
        context_parts = []
        
        # 1. State Anchor (identity context)
        anchor_text = self.get_state_anchor_for_prompt()
        if anchor_text:
            context_parts.append("## Identity Context")
            context_parts.append(anchor_text[:max_chars // 3])
        
        # 2. Relevant Golden Moments
        golden = self.retrieve_golden_moments(query, top_k=3)
        if golden:
            context_parts.append("\n## Preserved Memories (Golden Moments)")
            for gm in golden:
                text = gm.raw_text[:200] + "..." if len(gm.raw_text) > 200 else gm.raw_text
                context_parts.append(f"- {text}")
        
        # 3. Fractal Tree Drill-Down
        if self.fractal_tree:
             tree_context = self._traverse_fractal_tree(query, max_chars=max_chars // 2)
             if tree_context:
                 context_parts.append("\n## Fractal Archive Context")
                 context_parts.append(tree_context)
        
        return "\n".join(context_parts)

    def _traverse_fractal_tree(self, query: str, max_chars: int = 1000) -> str:
        """
        Traverse the fractal tree to find relevant context.
        
        Strategy:
        1. Check Epoch summaries.
        2. If an Epoch is relevant, check its Arcs.
        3. If an Arc is relevant, check its Scenes (summary only).
        """
        if not FRACTAL_MEMORY_AVAILABLE or not self.fractal_tree:
            return ""
            
        relevant_context = []
        chars_used = 0
        
        # Simple keyword overlap for now (can be upgraded to vector search)
        query_terms = set(query.lower().split())
        
        # Access all epochs (assuming loaded in memory or via graph - simplistic check for now)
        # Note: In a real implementation, we'd fetch nodes by ID.
        # For now, we assume we can't easily fetch ALL nodes without a graph traversal helper.
        # But we do have graph_data_manager.
        
        # Since we haven't implemented full graph synchronization for the Tree yet,
        # we will placeholder this with a comment.
        # Story 2.1 is mainly about Schema. The Traversal requires the graph to be populated.
        
        # Fallback: Just return the root summary
        if self.fractal_tree.summary:
            relevant_context.append(f"Life Summary: {self.fractal_tree.summary}")
            
        return "\n".join(relevant_context)

