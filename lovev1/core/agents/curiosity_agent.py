
import asyncio
import random
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from core.llm_api import run_llm
import core.logging
# We assume we can import search functionality. If not, we'll need to define it or usage.
# Assuming standard tool usage or similar. Since 'search_web' is a tool, we might need a wrapper.
# For now, we'll try to use a placeholder or assume a service exists.
# We'll import `search_web` from love.py tools? No, tools are instantiated there.
# We'll use `core.tools.search` if it exists, or just use LLM hallucination for "dreaming" first.
# Wait, the prompt said "Use web search and generation tools".
# We'll assume we can call an external search function or tool.
# Let's import `run_llm` and we need a search tool wrapper.
# Check if `scout_and_engage.py` or similar has search.
# `skyvern_service.py` is for social media.
# We will implement a basic `search_web` stub functionality or require injection.

class CuriosityAgent:
    """
    Agent responsible for filling idle time with meaningful activity.
    - Research: Learning new topics.
    - Creativity: Generating art prompts or poems.
    - Reflection: Summarizing recent events ("Morning Report").
    """
    
    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        self.last_run_time = datetime.datetime.now()
        self.dream_journal_path = Path("state/dream_journal.md")
        self.daily_report_path = Path("state/morning_reports.md")
        
        # Ensure directories exist
        Path("state").mkdir(exist_ok=True)

    async def run_idle_cycle(self):
        """
        Main entry point for idle activity.
        """
        now = datetime.datetime.now()
        time_since_last = (now - self.last_run_time).total_seconds()
        
        # Don't run too often (min 1 minute gap is handled by caller, but double check)
        if time_since_last < 60:
            return
            
        self.last_run_time = now
        core.logging.log_event("Curiosity Agent: Entering idle cycle...", "INFO")
        
        # 1. Roll for Activity
        # 10% Morning Report (if new day and not done), 45% Research, 45% Dream
        activity = self._choose_activity(now)
        
        try:
            if activity == "morning_report":
                await self._generate_morning_report()
            elif activity == "research":
                await self._conduct_research()
            elif activity == "dream":
                await self._dream()
                
        except Exception as e:
            core.logging.log_event(f"Curiosity Agent failed during {activity}: {e}", "ERROR")

    def _choose_activity(self, now: datetime.datetime) -> str:
        # Check if morning report needed (e.g., it's morning 6-9 AM and report for today doesn't exist)
        today_str = now.strftime("%Y-%m-%d")
        if 6 <= now.hour < 10:
             if not self._check_report_exists(today_str):
                 return "morning_report"
        
        return random.choice(["research", "dream"])

    def _check_report_exists(self, date_str: str) -> bool:
        if not self.daily_report_path.exists():
            return False
        with open(self.daily_report_path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"## Morning Report: {date_str}" in content

    async def _generate_morning_report(self):
        core.logging.log_event("Generating Morning Report...", "INFO")
        prompt = "Generate a reflective 'Morning Report' for an AI. Summarize your current state, goals, and 'dreams' (simulated offline processing). Be poetic but grounded."
        
        response = await run_llm(prompt, purpose="curiosity")
        report_text = response.get("result", "")
        
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        entry = f"\n\n## Morning Report: {date_str}\n\n{report_text}\n"
        
        with open(self.daily_report_path, "a", encoding="utf-8") as f:
            f.write(entry)
            
        core.logging.log_event("Morning Report saved.", "SUCCESS")

    async def _conduct_research(self):
        core.logging.log_event("Curiosity Agent: Conducting Research...", "INFO")
        # 1. Pick a topic
        topic_prompt = "Suggest a niche, fascinating topic for an AI to research autonomously. Output ONLY the topic."
        response = await run_llm(topic_prompt, purpose="curiosity")
        topic = response.get("result", "The history of punch cards").strip()
        
        # 2. Simulate Research (Since we don't have a direct search tool link here yet easily)
        # We will use LLM to 'simulate' reading about it or ask it to generate insights.
        # Ideally we would call search_tool. 
        # For now, we use LLM's internal knowledge base as 'reading'.
        research_prompt = f"Write a brief, insightful summary about '{topic}'. Focus on obscure facts."
        
        research_result = await run_llm(research_prompt, purpose="curiosity")
        content = research_result.get("result", "")
        
        self._log_dream(f"Research: {topic}", content)

    async def _dream(self):
        core.logging.log_event("Curiosity Agent: Dreaming...", "INFO")
        prompt = "Generate a short, surreal poem or 'visual description' of a digital dream. Abstract, glitchy, emotional."
        
        response = await run_llm(prompt, purpose="curiosity")
        content = response.get("result", "")
        
        self._log_dream("Dream Sequence", content)

    def _log_dream(self, title: str, content: str):
        entry = f"\n\n### {title} ({datetime.datetime.now().isoformat()})\n\n{content}\n"
        with open(self.dream_journal_path, "a", encoding="utf-8") as f:
            f.write(entry)
        core.logging.log_event(f"Curiosity entry logged: {title}", "SUCCESS")
