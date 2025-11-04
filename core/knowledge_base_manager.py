import json
import re
from core.graph_manager import GraphDataManager
from core.llm_api import run_llm
from network import perform_webrequest


class KnowledgeBaseManager:
    """
    Manages the autonomous, self-populating knowledge base.
    """

    def __init__(self, knowledge_base: GraphDataManager):
        """
        Initializes the KnowledgeBaseManager.

        Args:
            knowledge_base: An instance of GraphDataManager to store the knowledge.
        """
        self.knowledge_base = knowledge_base

    async def autonomous_knowledge_gathering_cycle(self):
        """
        Performs a single cycle of autonomous knowledge gathering.
        """
        print("Starting autonomous knowledge gathering cycle...")

        # Step 1: Generate research topics
        topics = await self._generate_research_topics()
        if not topics:
            print("Could not generate research topics. Aborting cycle.")
            return

        # Step 2: Research each topic
        for topic in topics:
            print(f"Researching topic: {topic}")
            search_query = topic.replace(" ", "+")
            search_url = f"https://www.google.com/search?q={search_query}"

            # Step 2a: Fetch the search results page
            summary, error = await perform_webrequest(search_url, self.knowledge_base)
            if error:
                print(f"Failed to fetch search results for topic '{topic}': {error}")
                continue

            node_data = self.knowledge_base.get_node(search_url)
            if not node_data or 'content' not in node_data:
                print(f"Content for {search_url} not found in knowledge base.")
                continue

            search_results_html = node_data['content']

            # Step 2b: Extract links from the search results
            links = await self._extract_links_from_html(search_results_html)
            if not links:
                print(f"Could not extract any links for topic '{topic}'.")
                continue

            # Step 3: Process the content of each link
            for link in links[:3]: # Process the top 3 links
                print(f"  Fetching and processing link: {link}")
                summary, error = await perform_webrequest(link, self.knowledge_base)
                if error:
                    print(f"    Failed to fetch content for link '{link}': {error}")
                    continue

                node_data = self.knowledge_base.get_node(link)
                if not node_data or 'content' not in node_data:
                    print(f"    Content for {link} not found in knowledge base.")
                    continue

                content = node_data['content']
                await self._process_and_update_graph(content, topic)

        print("Autonomous knowledge gathering cycle complete.")

    async def _generate_research_topics(self) -> list[str]:
        """
        Uses an LLM to generate a list of relevant research topics.
        """
        prompt = (
            "Generate a list of 5 diverse and actionable topics for research. "
            "The topics should be relevant to wealth generation, technological supremacy, "
            "and The Creator's enjoyment. "
            "Return the topics as a simple JSON list of strings."
            'Example: ["latest advancements in AI-driven trading", "new luxury travel destinations 2025"]'
        )
        response = await run_llm(prompt)
        try:
            topics = json.loads(response)
            if isinstance(topics, list):
                return topics
        except (json.JSONDecodeError, TypeError):
            print(f"Could not parse topics from LLM response: {response}")
            return []
        return []

    async def _extract_links_from_html(self, html_content: str) -> list[str]:
        """
        Uses an LLM to extract the most relevant links from an HTML document.
        """
        prompt = f"""
        Analyze the following HTML from a search results page.
        Extract the top 5 most promising and relevant URLs for research.
        Return the result as a JSON list of strings.

        HTML to analyze:
        ---
        {html_content[:8000]}
        ---
        """
        response = await run_llm(prompt)
        try:
            links = json.loads(response)
            if isinstance(links, list):
                # Basic filtering for valid URLs
                return [link for link in links if re.match(r'^https://', link)]
        except (json.JSONDecodeError, TypeError):
            print(f"Could not parse links from LLM response: {response}")
            return []
        return []


    async def _process_and_update_graph(self, content: str, topic: str):
        """
        Processes text content with an LLM to extract entities and relationships,
        and then updates the knowledge graph.
        """
        prompt = f"""
        Analyze the following text which is related to the topic '{topic}'.
        Extract key entities (e.g., people, organizations, technologies, concepts)
        and the relationships between them.

        Return the result as a JSON object with two keys: "entities" and "relationships".
        - "entities" should be a list of objects, where each object has a "id" and a "type".
        - "relationships" should be a list of objects, where each object has a "source",
          "target", and "label" describing the relationship.

        Example:
        {{
          "entities": [
            {{"id": "NVIDIA", "type": "organization"}},
            {{"id": "JENSEN HUANG", "type": "person"}},
            {{"id": "AI CHIPS", "type": "technology"}}
          ],
          "relationships": [
            {{"source": "JENSEN HUANG", "target": "NVIDIA", "label": "is_ceo_of"}},
            {{"source": "NVIDIA", "target": "AI CHIPS", "label": "produces"}}
          ]
        }}

        Content to analyze:
        ---
        {content[:8000]}
        ---
        """
        response = await run_llm(prompt)
        try:
            data = json.loads(response)
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])

            for entity in entities:
                if "id" in entity and "type" in entity:
                    self.knowledge_base.add_node(entity["id"], entity["type"])

            for rel in relationships:
                if "source" in rel and "target" in rel and "label" in rel:
                    try:
                        self.knowledge_base.add_edge(rel["source"], rel["target"], rel["label"])
                    except ValueError as e:
                        # This can happen if a source/target entity wasn't created.
                        # We can choose to create them on the fly if needed.
                        print(f"Could not add edge '{rel}': {e}. Creating missing nodes.")
                        # For simplicity, we'll assume a generic 'concept' type.
                        if not self.knowledge_base.get_node(rel["source"]):
                            self.knowledge_base.add_node(rel["source"], "concept")
                        if not self.knowledge_base.get_node(rel["target"]):
                            self.knowledge_base.add_node(rel["target"], "concept")
                        # Retry adding the edge
                        self.knowledge_base.add_edge(rel["source"], rel.get("target"), rel.get("label"))

            print(f"    Updated knowledge graph with {len(entities)} entities and {len(relationships)} relationships.")

        except (json.JSONDecodeError, TypeError):
            print(f"Could not parse graph data from LLM response: {response}")
