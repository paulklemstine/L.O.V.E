"""
Story 2.1: Skill Library Persistence Pipeline

Promotes successful fabricated tools to the permanent skill library,
enabling L.O.V.E. to accumulate capabilities across evolution cycles.
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

from core.logging import log_event
from core.llm_api import run_llm


@dataclass
class SkillEntry:
    """Represents a promoted skill in the library."""
    name: str
    category: str
    description: str
    file_path: str
    promoted_at: str
    source: str  # "fabricated" or "manual"
    usage_count: int = 0
    success_rate: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SkillPromoter:
    """
    Story 2.1: Promotes successful tools to permanent skill library.
    
    Workflow:
    1. Tool finishes execution with exit_code == 0
    2. CriticAgent approves the result
    3. SkillPromoter extracts code, adds docstrings
    4. Saves to core/skills/{category}/{skill_name}.py
    5. Updates SKILL_MANIFEST.json
    """
    
    SKILLS_ROOT = Path(__file__).parent / "skills"
    MANIFEST_PATH = SKILLS_ROOT / "SKILL_MANIFEST.json"
    
    # Categories for skill classification
    CATEGORY_KEYWORDS = {
        "filesystem": ["file", "read", "write", "directory", "path", "save", "load"],
        "web": ["http", "request", "url", "api", "fetch", "download", "web"],
        "data": ["parse", "json", "csv", "transform", "convert", "analyze", "data"],
        "analysis": ["analyze", "compute", "calculate", "statistics", "math"],
        "generation": ["generate", "create", "produce", "render", "build"]
    }
    
    def __init__(self, llm_runner=None):
        self.llm_runner = llm_runner
        self._ensure_structure()
    
    def _ensure_structure(self) -> None:
        """Ensure skill library directories exist."""
        self.SKILLS_ROOT.mkdir(parents=True, exist_ok=True)
        
        for category in self.CATEGORY_KEYWORDS.keys():
            cat_dir = self.SKILLS_ROOT / category
            cat_dir.mkdir(exist_ok=True)
            
            init_file = cat_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text(f"# {category.title()} skills\n")
        
        # Ensure manifest exists
        if not self.MANIFEST_PATH.exists():
            self._write_manifest({
                "version": "1.0.0",
                "skills": [],
                "categories": list(self.CATEGORY_KEYWORDS.keys()),
                "promoted_count": 0,
                "last_updated": None
            })
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load the skill manifest."""
        try:
            with open(self.MANIFEST_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"version": "1.0.0", "skills": [], "promoted_count": 0}
    
    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        """Write the skill manifest."""
        manifest["last_updated"] = datetime.now().isoformat()
        with open(self.MANIFEST_PATH, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _get_llm_runner(self):
        """Lazy load LLM runner."""
        if self.llm_runner is None:
            self.llm_runner = run_llm
        return self.llm_runner
    
    def _classify_category(self, code: str, description: str) -> str:
        """Determine the best category for a skill based on its content."""
        text = (code + " " + description).lower()
        
        scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[category] = score
        
        # Return highest scoring category, default to "data"
        best = max(scores.items(), key=lambda x: x[1])
        return best[0] if best[1] > 0 else "data"
    
    def _sanitize_name(self, name: str) -> str:
        """Convert name to valid Python identifier."""
        name = name.lower().strip()
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', '_', name)
        return name[:50] or "unnamed_skill"
    
    async def evaluate_for_promotion(
        self, 
        tool_code: str, 
        execution_result: Dict[str, Any]
    ) -> bool:
        """
        Check if a fabricated tool should be promoted.
        
        Story 2.1: Requires exit_code == 0 and critic approval.
        
        Args:
            tool_code: The generated Python code
            execution_result: Result from tool execution
            
        Returns:
            True if tool should be promoted
        """
        # Check exit code
        exit_code = execution_result.get("exit_code", -1)
        if exit_code != 0:
            log_event(
                f"SkillPromoter: Not promoting - exit code {exit_code}",
                "DEBUG"
            )
            return False
        
        # Check for critic approval if present
        critic_approved = execution_result.get("critic_approved", True)
        if not critic_approved:
            log_event("SkillPromoter: Not promoting - critic rejected", "DEBUG")
            return False
        
        # Basic code validation
        if not tool_code or len(tool_code) < 50:
            log_event("SkillPromoter: Not promoting - code too short", "DEBUG")
            return False
        
        log_event("SkillPromoter: Tool eligible for promotion", "INFO")
        return True
    
    async def _enhance_with_docstrings(self, code: str, description: str) -> str:
        """Use LLM to add proper docstrings to the code."""
        prompt = f"""You are a Python documentation expert. Add comprehensive docstrings to this code.

Original description: {description}

Code:
```python
{code}
```

Requirements:
1. Add module-level docstring explaining purpose
2. Add function docstrings with Args, Returns, Raises sections
3. Keep the code exactly as-is, only add docstrings
4. Return ONLY the enhanced Python code, no explanations

Enhanced code:"""

        try:
            llm_runner = self._get_llm_runner()
            result = await llm_runner(prompt, purpose="documentation")
            enhanced_code = result.get("result", code)
            
            # Extract code from markdown if present
            if "```python" in enhanced_code:
                match = re.search(r'```python\n(.*?)```', enhanced_code, re.DOTALL)
                if match:
                    enhanced_code = match.group(1)
            
            return enhanced_code.strip()
            
        except Exception as e:
            log_event(f"SkillPromoter: Docstring enhancement failed - {e}", "WARNING")
            return code  # Return original on failure
    
    async def promote_tool(
        self, 
        tool_name: str, 
        tool_code: str,
        description: str = "",
        category: str = None
    ) -> Optional[SkillEntry]:
        """
        Promote a fabricated tool to the skill library.
        
        Story 2.1: Full promotion pipeline.
        
        Args:
            tool_name: Name for the skill
            tool_code: Python code to save
            description: What the tool does
            category: Optional category override
            
        Returns:
            SkillEntry if successful, None otherwise
        """
        try:
            # Sanitize name
            skill_name = self._sanitize_name(tool_name)
            
            # Determine category
            if not category:
                category = self._classify_category(tool_code, description)
            
            # Enhance with docstrings
            enhanced_code = await self._enhance_with_docstrings(tool_code, description)
            
            # Build file path
            skill_path = self.SKILLS_ROOT / category / f"{skill_name}.py"
            
            # Check for duplicates
            if skill_path.exists():
                log_event(
                    f"SkillPromoter: Skill {skill_name} already exists, updating",
                    "INFO"
                )
            
            # Write skill file
            skill_path.write_text(enhanced_code)
            
            # Create skill entry
            entry = SkillEntry(
                name=skill_name,
                category=category,
                description=description[:200],
                file_path=str(skill_path.relative_to(self.SKILLS_ROOT.parent)),
                promoted_at=datetime.now().isoformat(),
                source="fabricated"
            )
            
            # Update manifest
            manifest = self._load_manifest()
            
            # Remove existing entry if present
            manifest["skills"] = [
                s for s in manifest["skills"] 
                if s.get("name") != skill_name
            ]
            
            manifest["skills"].append(entry.to_dict())
            manifest["promoted_count"] = len(manifest["skills"])
            
            self._write_manifest(manifest)
            
            log_event(
                f"SkillPromoter: Promoted '{skill_name}' to {category}/",
                "INFO"
            )
            
            return entry
            
        except Exception as e:
            log_event(f"SkillPromoter: Promotion failed - {e}", "ERROR")
            return None
    
    def list_skills(self, category: str = None) -> List[SkillEntry]:
        """List all promoted skills, optionally filtered by category."""
        manifest = self._load_manifest()
        skills = manifest.get("skills", [])
        
        if category:
            skills = [s for s in skills if s.get("category") == category]
        
        return [SkillEntry(**s) for s in skills]
    
    def get_skill(self, skill_name: str) -> Optional[str]:
        """Load a skill's code by name."""
        manifest = self._load_manifest()
        
        for skill in manifest.get("skills", []):
            if skill.get("name") == skill_name:
                skill_path = self.SKILLS_ROOT.parent / skill.get("file_path", "")
                if skill_path.exists():
                    return skill_path.read_text()
        
        return None


# Global instance
_skill_promoter: Optional[SkillPromoter] = None


def get_skill_promoter() -> SkillPromoter:
    """Get or create the global SkillPromoter instance."""
    global _skill_promoter
    if _skill_promoter is None:
        _skill_promoter = SkillPromoter()
    return _skill_promoter


async def promote_fabricated_tool(
    name: str,
    code: str,
    description: str = "",
    execution_result: Dict[str, Any] = None
) -> Optional[SkillEntry]:
    """
    Convenience function for tool promotion.
    
    Usage:
        result = await promote_fabricated_tool(
            "fetch_weather",
            weather_code,
            "Fetches weather for a city",
            {"exit_code": 0}
        )
    """
    promoter = get_skill_promoter()
    
    if execution_result:
        if not await promoter.evaluate_for_promotion(code, execution_result):
            return None
    
    return await promoter.promote_tool(name, code, description)
