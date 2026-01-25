import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List

from core.intent_layer.models import IntentNode, INTENT_FILENAME
from core.token_utils import count_tokens_for_api_models

logger = logging.getLogger(__name__)

class IntentLoader:
    """
    Service to traverse the file system and build the Intent Layer graph.
    """

    @staticmethod
    @lru_cache(maxsize=128)
    def get_context_stack(target_path: str) -> List[IntentNode]:
        """
        Traverses upward from target_path to root, collecting AGENTS.md files.
        Returns a list ordered from ROOT -> LEAF (Broadest -> Specific).
        
        Args:
            target_path: The absolute path to start searching from. Can be a file or directory.
            
        Returns:
            List of IntentNode objects properly ordered.
        """
        # Ensure we have an absolute path
        path = Path(target_path).resolve()
        nodes = []
        
        # If target_path is a file, start from its parent directory
        if path.is_file():
            current = path.parent
        else:
            current = path

        # Traverse upwards
        while True:
            intent_file = current / INTENT_FILENAME
            
            if intent_file.exists() and intent_file.is_file():
                try:
                    content = intent_file.read_text(encoding='utf-8')
                    # Basic parsing could happen here, or we treat it as raw content for now
                    # Story 1 mentioned summary parsing, but we can iterate on that.
                    
                    node = IntentNode(
                        path=str(current),
                        content=content,
                        summary=None, # To be implemented in Story 4/Parsing logic
                    )
                    nodes.append(node)
                except Exception as e:
                    logger.warning(f"Failed to read Intent Layer file at {intent_file}: {e}")
            
            # Check if we have reached the root of the file system
            if current.parent == current:
                break
                
            current = current.parent

        # The traversal goes Leaf -> Root, but we want Root -> Leaf (Context stack)
        return list(reversed(nodes))

    @staticmethod
    def format_context_stack(stack: List[IntentNode]) -> str:
        """
        Formats the context stack into a string for injection into the system prompt.
        """
        if not stack:
            return ""

        output = ["*** INTENT LAYER CONTEXT ***"]
        
        for node in stack:
            # Determine label based on position
            # This is a bit loose since we don't know the full graph, but relative to the stack:
            # First is ROOT, Last is CURRENT/LEAF, others are PARENT
            
            label = "PARENT"
            if node == stack[0]:
                label = "ROOT"
            if node == stack[-1]:
                label = "CURRENT"
                
            # If there's only one node, it's both ROOT and CURRENT, but ROOT takes precedence in naming usually, 
            # or we can just say [ROOT/CURRENT]
            if len(stack) == 1:
                label = "ROOT/CURRENT"
            
            # Format: [LABEL] path:
            # ... content ...
            
            output.append(f"\n[{label}] {node.path}:")
            # TODO: Use summary if available and not CURRENT/ROOT (Story 4)
            output.append(node.content.strip())
            output.append("---")
            
        return "\n".join(output)

    @staticmethod
    def compress_context(stack: List[IntentNode], max_tokens: int) -> List[IntentNode]:
        """
        Reduces the token footprint of the context stack to fit within max_tokens.
        Prioritizes Leaf (Current) and Root nodes.
        Compresses intermediate nodes by keeping only key sections.
        """
        if not stack:
            return []

        # Calculate costs
        # usage: [(index, node, count)]
        usage = []
        total_tokens = 0
        for i, node in enumerate(stack):
            count = count_tokens_for_api_models(node.content)
            usage.append({'index': i, 'node': node, 'count': count})
            total_tokens += count
            
        if total_tokens <= max_tokens:
            return stack
            
        # We need to compress.
        # Strategy:
        # 1. Keep Leaf (last) full.
        # 2. Keep Root (first) full (or as much as possible).
        # 3. Compress intermediates.
        
        leaf_idx = len(stack) - 1
        root_idx = 0
        
        leaf_tokens = usage[leaf_idx]['count']
        root_tokens = usage[root_idx]['count'] if len(stack) > 1 else 0
        
        # Budget for intermediates
        remaining_budget = max_tokens - leaf_tokens - root_tokens
        
        # If we are already over budget with just Root + Leaf
        if remaining_budget < 0:
            logger.warning(f"Intent Layer budget ({max_tokens}) too small for Root+Leaf ({leaf_tokens + root_tokens}). Truncating Root.")
            # Keep Leaf full, truncate Root
            # New Root budget = max_tokens - leaf_tokens (could be negative if Leaf is huge, handle that)
            root_budget = max(0, max_tokens - leaf_tokens)
            
            # Truncate Root content
            root_node = stack[root_idx]
            truncated_root = IntentNode(
                path=root_node.path,
                content=root_node.content[:root_budget * 4], # Approx chars
                summary=root_node.summary,
                parent=root_node.parent,
                downlinks=root_node.downlinks
            )
            
            new_stack = [truncated_root]
            if len(stack) > 1:
                new_stack.append(stack[leaf_idx])
            return new_stack

        # We have budget for intermediates (or at least 0)
        # Intermediates are indices 1 to len-2
        intermediates = usage[1:-1]
        
        # Simple distribution: divide remaining budget equally? 
        # Or better: Extract key sections from intermediates.
        
        if not intermediates:
            return stack # Should have been covered by total_tokens check unless logic flaw
            
        # Compress intermediates
        compressed_intermediates = []
        token_per_intermediate = max(100, remaining_budget // len(intermediates)) # Ensure at least some budget
        
        for item in intermediates:
            node = item['node']
            compressed_content = IntentLoader._extract_key_sections(node.content, token_per_intermediate)
            
            compressed_node = IntentNode(
                path=node.path,
                content=compressed_content, # Modified content
                summary=node.summary,
                parent=node.parent,
                downlinks=node.downlinks
            )
            compressed_intermediates.append(compressed_node)
            
        # Reconstruct stack
        final_stack = [stack[0]] + compressed_intermediates + [stack[-1]]
        return final_stack

    @staticmethod
    def _extract_key_sections(content: str, max_tokens: int) -> str:
        """
        Extracts 'Purpose', 'Invariants', 'Anti-patterns' headers and truncates to fit budget.
        """
        # Simple line-based filtering for now, or just truncate
        # To be robust, we'd parse markdown. 
        # For this turn: prioritized truncation.
        
        important_headers = ["purpose", "invariant", "boundary", "anti-pattern", "constraint"]
        lines = content.split('\n')
        kept_lines = []
        
        current_section_important = False
        
        for line in lines:
            line_lower = line.lower()
            if line.startswith('#'):
                # Check header importance
                current_section_important = any(h in line_lower for h in important_headers)
                kept_lines.append(line)
            elif current_section_important:
                kept_lines.append(line)
            # else: skip unimportant sections for intermediates
            
        result = "\n".join(kept_lines)
        
        # Final truncate if still too big
        if count_tokens_for_api_models(result) > max_tokens:
             # Approximate char limit
             return result[:max_tokens * 4] + "\n...[truncated]"
             
        if not result.strip(): # If we filtered everything, keep regular truncation of original
             return content[:max_tokens * 4] + "\n...[truncated]"
             
        return result

