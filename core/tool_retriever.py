from typing import List, Any, Optional
import logging
from langchain_core.tools import BaseTool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

class ToolRetriever:
    def __init__(self, tools: List[BaseTool], index_path: str = "tool_index", validate_against_registry: bool = True):
        self.tools = tools
        self.index_path = index_path
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if validate_against_registry:
            self._validate_against_registry()
        
        self.vectorstore = self._initialize_vectorstore()
    
    def _validate_against_registry(self) -> None:
        """
        Validates that all tools exist in the global ToolRegistry.
        
        Logs a warning for any tool that exists in the retriever but not
        in the registry, as this indicates a potential "ghost tool".
        """
        try:
            from core.tool_registry import get_global_registry
            registry = get_global_registry()
            registry_tool_names = set(registry.list_tools())
            
            for tool in self.tools:
                tool_name = getattr(tool, 'name', str(tool))
                if tool_name not in registry_tool_names:
                    logging.warning(
                        f"ToolRetriever: Tool '{tool_name}' not found in ToolRegistry. "
                        "This may indicate a 'ghost tool' that exists in code but not in LLM context."
                    )
        except ImportError as e:
            logging.warning(f"ToolRetriever: Could not import tool_registry for validation: {e}")
        except Exception as e:
            logging.warning(f"ToolRetriever: Could not validate against registry: {e}")

    def _initialize_vectorstore(self):
        if not self.tools:
            return None
            
        texts = [f"{tool.name}: {tool.description}" for tool in self.tools]
        metadatas = [{"name": tool.name} for tool in self.tools]
        
        # In a real scenario, we might want to load from disk if exists, 
        # but for tools which might change, rebuilding in memory is often fast enough.
        return FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)

    def query_tools(self, query: str, k: int = 5) -> List[BaseTool]:
        if not self.vectorstore:
            return []
            
        docs = self.vectorstore.similarity_search(query, k=k)
        retrieved_tools = []
        for doc in docs:
            tool_name = doc.metadata["name"]
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                retrieved_tools.append(tool)
        return retrieved_tools

    def add_tools(self, new_tools: List[BaseTool]):
        """Adds new tools to the retriever."""
        self.tools.extend(new_tools)
        texts = [f"{tool.name}: {tool.description}" for tool in new_tools]
        metadatas = [{"name": tool.name} for tool in new_tools]
        self.vectorstore.add_texts(texts, metadatas=metadatas)

