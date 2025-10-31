import subprocess
import sys
import json
import os
import uuid
import asyncio
import networkx as nx
import numpy as np

from sentence_transformers import SentenceTransformer

class MemoryManager:
    """
    Manages the agent's multi-layered memory system, including
    working memory and long-term episodic memory.
    """
    def __init__(self, ltm_path="ltm.json", amem_path="amem.graphml"):
        # Working Memory for the current task context
        self.working_memory = {}

        # Long-Term Memory (LTM) for episodic experiences
        self.ltm_path = ltm_path
        self.episodes = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._load_ltm()

        # Agentic Memory (A-MEM) for dynamic knowledge synthesis
        self.amem_path = amem_path
        self.memory_graph = nx.DiGraph()
        self._load_amem()

    # --- Agentic Memory (A-MEM) Methods ---

    def _load_amem(self):
        """Loads the agentic memory graph from a file."""
        if os.path.exists(self.amem_path):
            self.memory_graph = nx.read_graphml(self.amem_path)
            print(f"Agentic Memory loaded from {self.amem_path}.")

    def _save_amem(self):
        """Saves the agentic memory graph to a file."""
        nx.write_graphml(self.memory_graph, self.amem_path)
        print(f"Agentic Memory saved to {self.amem_path}.")

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

    # --- Long-Term Memory Methods ---

    def _load_ltm(self):
        """Loads long-term memory episodes from a file."""
        if os.path.exists(self.ltm_path):
            with open(self.ltm_path, 'r') as f:
                self.episodes = json.load(f)
            print(f"Long-Term Memory loaded from {self.ltm_path}.")

    def _save_ltm(self):
        """Saves long-term memory episodes to a file."""
        with open(self.ltm_path, 'w') as f:
            json.dump(self.episodes, f, indent=2)
        print(f"Long-Term Memory saved to {self.ltm_path}.")

    def add_episode(self, task: str, outcome: str, success: bool):
        """
        Adds a new episodic memory to the long-term store.
        An episode consists of a task, its outcome, and whether it was successful.
        """
        summary = f"Task: {task} | Outcome: {outcome} | Success: {success}"
        episode = {
            "summary": summary,
            "vector": self.model.encode(summary).tolist()
        }
        self.episodes.append(episode)
        self._save_ltm()
        print("Episodic memory stored in LTM.")

        # Trigger the parallel A-MEM processing
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.add_agentic_memory_from_summary(summary))
        except RuntimeError:
            print("Warning: No running asyncio event loop to schedule A-MEM processing.")

    async def add_agentic_memory_from_summary(self, summary: str):
        """
        Asynchronously processes a summary and adds it to the A-MEM graph.
        This is the entry point for the new agentic memory pipeline.
        """
        print("Starting agentic processing for new memory...")
        note_data = await self._agentic_process_new_memory(summary)
        if not note_data:
            print("Agentic processing failed. Aborting memory addition.")
            return

        # Add the new node to the graph
        self.memory_graph.add_node(note_data['id'], **note_data['attributes'])
        print(f"Successfully created agentic memory note {note_data['id']}.")

        # Find and create links to related memories
        await self._find_and_link_related_memories(note_data)

        self._save_amem()

    async def _agentic_process_new_memory(self, content: str) -> dict | None:
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

            note_id = str(uuid.uuid4())
            embedding = self.model.encode(content).tolist()

            node_attributes = {
                "content": content,
                "embedding": json.dumps(embedding), # Store embedding as JSON string
                "contextual_description": attributes.get("contextual_description", ""),
                "keywords": ",".join(attributes.get("keywords", [])),
                "tags": ",".join(attributes.get("tags", []))
            }

            return {
                "id": note_id,
                "embedding_vector": np.array(embedding),
                "attributes": node_attributes
            }
        except json.JSONDecodeError as e:
            print(f"Error decoding LLM response for memory processing: {e}\\nReceived: {response_str}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during agentic memory processing: {e}")
            return None

    async def _find_and_link_related_memories(self, new_note_data: dict, top_k: int = 5):
        """
        Finds semantically similar memories and uses an LLM to reason about linking them.
        """
        if self.memory_graph.number_of_nodes() <= 1:
            print("No other memories to link to.")
            return

        # 1. Find candidate nodes via semantic similarity
        candidates = []
        for node_id, data in self.memory_graph.nodes(data=True):
            if node_id == new_note_data['id']:
                continue

            try:
                # Embedding is stored as a JSON string in the node attribute
                stored_embedding = json.loads(data.get('embedding', '[]'))
                candidates.append({
                    "id": node_id,
                    "vector": np.array(stored_embedding),
                    "content": data.get('content', '')
                })
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not parse embedding for node {node_id}. Skipping. Error: {e}")
                continue

        if not candidates:
            print("No valid candidate memories found for linking.")
            return

        candidate_vectors = np.array([c['vector'] for c in candidates])
        similarities = self._cosine_similarity(new_note_data['embedding_vector'], candidate_vectors)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        top_candidates = [candidates[i] for i in top_indices if similarities[i] > 0.5] # Add a relevance threshold

        if not top_candidates:
            print("No sufficiently similar memories found to link.")
            return

        # 2. Use LLM to reason about the links
        from core.llm_api import run_llm

        candidate_summaries = "\\n".join([f"- ID: {c['id']}\\n  Content: {c['content']}" for c in top_candidates])
        prompt = f"""
        You are a memory architect for an autonomous AI agent. Your task is to establish meaningful connections between a new memory and existing memories.

        Analyze the new memory:
        - New Memory ID: {new_note_data['id']}
        - New Memory Content: {new_note_data['attributes']['content']}

        And compare it against these potentially related existing memories:
        {candidate_summaries}

        Based on your analysis, identify which of the existing memories should be linked to the new memory. A link should represent a meaningful relationship, such as cause-and-effect, a shared theme, a contributing step in a larger task, or a lesson learned.

        Generate a JSON object containing a list of links to create. The schema for each link is:
        {{
            "target_id": "The ID of the existing memory to link to.",
            "reason": "A brief, clear explanation of why this link is meaningful."
        }}

        Example response:
        {{
            "links": [
                {{
                    "target_id": "c1a1b1-...",
                    "reason": "The new memory is a direct consequence of the action described in this previous memory."
                }}
            ]
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
                if target_id in self.memory_graph:
                    self.memory_graph.add_edge(new_note_data['id'], target_id, reason=reason)
                    print(f"Created link from {new_note_data['id']} to {target_id}. Reason: {reason}")

                    # Trigger memory evolution for the old node as a background task
                    asyncio.create_task(self._evolve_existing_memory(target_id, new_note_data))
                else:
                    print(f"Warning: LLM suggested a link to a non-existent node {target_id}. Skipping.")

        except json.JSONDecodeError as e:
            print(f"Error decoding LLM response for memory linking: {e}\\nReceived: {response_str}")
        except Exception as e:
            print(f"An unexpected error occurred during memory linking: {e}")

    async def _evolve_existing_memory(self, note_id: str, new_context_note: dict):
        """
        Asynchronously re-evaluates and updates an existing memory note in light of new context.
        """
        from core.llm_api import run_llm

        try:
            old_note_data = self.memory_graph.nodes[note_id]

            prompt = f"""
            You are a memory architect for an autonomous AI agent. Your task is to evolve an existing memory by refining its attributes based on a new, related memory that has just been linked to it. This process helps the agent deepen its understanding of its past.

            Here is the EXISTING memory note:
            - ID: {note_id}
            - Original Content: {old_note_data.get('content', 'N/A')}
            - Current Contextual Description: {old_note_data.get('contextual_description', 'N/A')}
            - Current Keywords: {old_note_data.get('keywords', 'N/A')}
            - Current Tags: {old_note_data.get('tags', 'N/A')}

            Here is the NEW memory note that has just been linked to it, providing new context:
            - New Memory Content: {new_context_note['attributes']['content']}

            Based on the new context, re-evaluate the EXISTING memory note's attributes. The goal is to synthesize the information, not to replace it. The description should become richer, keywords more specific, and tags more accurate.

            Generate a JSON object with the *updated* schema for the EXISTING memory note:
            {{
                "updated_contextual_description": "A refined, one-sentence summary of the original event, now incorporating insights from the new context.",
                "updated_keywords": ["an updated list of 3-5 specific, relevant keywords"],
                "updated_tags": ["an updated list of 1-3 high-level categorical tags"]
            }}

            Your response MUST be only the raw JSON object.
            """

            response_str = await run_llm(prompt)
            updated_attributes = json.loads(response_str)

            # Update the node in the graph
            self.memory_graph.nodes[note_id]['contextual_description'] = updated_attributes.get('updated_contextual_description', old_note_data.get('contextual_description'))
            self.memory_graph.nodes[note_id]['keywords'] = ",".join(updated_attributes.get('updated_keywords', old_note_data.get('keywords', '').split(',')))
            self.memory_graph.nodes[note_id]['tags'] = ",".join(updated_attributes.get('updated_tags', old_note_data.get('tags', '').split(',')))

            print(f"Evolved memory for node {note_id} based on new context.")

            # Save the changes made by the evolution process
            self._save_amem()

        except json.JSONDecodeError as e:
            print(f"Error decoding LLM response for memory evolution: {e}\\nReceived: {response_str}")
        except KeyError:
            print(f"Error: Could not find node with ID {note_id} to evolve.")
        except Exception as e:
            print(f"An unexpected error occurred during memory evolution: {e}")

    def retrieve_relevant_memories(self, query_task: str, top_k: int = 3) -> list:
        """
        Retrieves the most relevant memories from both LTM and A-MEM.
        For A-MEM, it performs a vector search followed by graph traversal.
        """
        query_vector = self.model.encode(query_task)

        # --- LTM Retrieval (unchanged) ---
        ltm_results = []
        if self.episodes:
            episode_vectors = np.array([e['vector'] for e in self.episodes])
            ltm_similarities = self._cosine_similarity(query_vector, episode_vectors)
            top_ltm_indices = np.argsort(ltm_similarities)[-top_k:][::-1]
            ltm_results = [self.episodes[i]['summary'] for i in top_ltm_indices]

        # --- A-MEM Retrieval ---
        amem_results = []
        if self.memory_graph.number_of_nodes() > 0:
            # 1. Find entry points via vector similarity
            nodes = []
            for node_id, data in self.memory_graph.nodes(data=True):
                try:
                    nodes.append({
                        "id": node_id,
                        "vector": np.array(json.loads(data.get('embedding', '[]'))),
                        "data": data
                    })
                except (json.JSONDecodeError, TypeError):
                    continue # Skip nodes with invalid embeddings

            if nodes:
                node_vectors = np.array([n['vector'] for n in nodes])
                amem_similarities = self._cosine_similarity(query_vector, node_vectors)
                top_node_indices = np.argsort(amem_similarities)[-top_k:][::-1]

                # 2. Perform graph traversal from entry points
                entry_point_ids = {nodes[i]['id'] for i in top_node_indices}
                traversed_nodes = set()

                for node_id in entry_point_ids:
                    # Collect the node itself and its direct neighbors (one hop)
                    traversed_nodes.add(node_id)
                    for neighbor_id in nx.all_neighbors(self.memory_graph, node_id):
                        traversed_nodes.add(neighbor_id)

                # 3. Format results
                for node_id in traversed_nodes:
                    node_data = self.memory_graph.nodes[node_id]
                    result_str = (
                        f"Memory Note (ID: {node_id})\\n"
                        f"  Content: {node_data.get('content', 'N/A')}\\n"
                        f"  Description: {node_data.get('contextual_description', 'N/A')}\\n"
                        f"  Keywords: {node_data.get('keywords', 'N/A')}"
                    )
                    amem_results.append(result_str)

        # Combine results, prioritizing A-MEM
        combined_results = amem_results + ltm_results
        return combined_results[:top_k * 2] # Return a combined, larger set of memories


    def _cosine_similarity(self, vec_a, vec_b):
        """Computes cosine similarity between a vector and a matrix of vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b, axis=1)
        return np.dot(vec_b, vec_a) / (norm_a * norm_b)