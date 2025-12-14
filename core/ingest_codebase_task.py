
import asyncio
import os
import aiofiles
from core.memory.memory_manager import MemoryManager
from core.logging import log_event

class IngestCodebaseTask:
    """
    Background task to scan the codebase and ingest file summaries into the Knowledge Base.
    """
    def __init__(self, memory_manager: MemoryManager, root_dir: str, interval_seconds: int = 3600):
        self.memory_manager = memory_manager
        self.root_dir = root_dir
        self.interval_seconds = interval_seconds
        self.running = False
        self.task = None

    async def start(self):
        self.running = True
        self.task = asyncio.create_task(self._run_loop())
        log_event("IngestCodebaseTask started.", "INFO")

    def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
        log_event("IngestCodebaseTask stopped.", "INFO")

    async def _run_loop(self):
        while self.running:
            try:
                await self.ingest_codebase()
                await asyncio.sleep(self.interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_event(f"Error in IngestCodebaseTask: {e}", "ERROR")
                await asyncio.sleep(60) # Retry after a minute on error

    async def ingest_codebase(self):
        log_event("Starting codebase ingestion...", "INFO")
        for root, dirs, files in os.walk(self.root_dir):
            if ".git" in dirs:
                dirs.remove(".git")
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
            
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    await self._process_file(filepath)
        log_event("Codebase ingestion complete.", "INFO")

    async def _process_file(self, filepath: str):
        try:
            relative_path = os.path.relpath(filepath, self.root_dir)
             # Check if we've already ingested this version? (Future optimization)
            
            async with aiofiles.open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = await f.read()

            summary = f"Codebase File: {relative_path}\n\nContent Preview:\n{content[:500]}..."
            tags = ["Codebase", "Documentation" if filepath.endswith(".md") else "SourceCode"]
            
            # Use add_episode to leverage the full pipeline (folding, embedding, etc.)
            await self.memory_manager.add_episode(summary, tags=tags)
            
        except Exception as e:
            log_event(f"Failed to ingest file {filepath}: {e}", "WARNING")
