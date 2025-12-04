import asyncio
import time
import uuid
import json
import os
import requests
import threading
from threading import Thread, RLock, Lock
from collections import deque
import traceback
import re
import shutil
import subprocess
import logging
import sys

# Ensure root directory is in path for sandbox import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from sandbox import Sandbox
except ImportError:
    # Fallback if running from root
    try:
        from sandbox import Sandbox
    except ImportError:
        logging.warning("Could not import Sandbox.")

import core.logging
from core.retry import retry
from core.llm_api import run_llm
from core.prompt_registry import get_prompt_registry
from utils import get_git_repo_info
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from core.desire_state import load_desire_state, get_current_desire, advance_to_next_desire, set_current_task_id_for_desire, clear_desire_state
from core.evolution_state import load_evolution_state, get_current_story, advance_to_next_story, clear_evolution_state, set_current_task_id
from openevolve import run_evolution
from core.openevolve_evaluator import evaluate_evolution

# --- JULES ASYNC TASK MANAGER ---
class JulesTaskManager:
    """
    Manages concurrent evolution tasks via the Jules API in a non-blocking way.
    It uses a background thread to poll for task status and merge PRs.
    """
    STUCK_TASK_TIMEOUT_SECONDS = 3600  # 1 hour

    def __init__(self, console, loop, deep_agent_engine=None, love_state=None, is_creator_instance=False, restart_callback=None, save_state_callback=None):
        self.console = console
        self.loop = loop
        self.deep_agent_engine = deep_agent_engine
        self.love_state = love_state if love_state is not None else {}
        self.is_creator_instance = is_creator_instance
        self.restart_callback = restart_callback
        self.save_state_callback = save_state_callback
        
        self.tasks = self.love_state.setdefault('love_tasks', {})
        self.completed_tasks = deque(self.love_state.setdefault('completed_tasks', []), maxlen=6)
        self.lock = RLock()
        self.max_concurrent_tasks = 5
        self.thread = Thread(target=self._task_loop, daemon=True)
        self.active = True

    def start(self):
        """Starts the background polling thread."""
        self.thread.start()
        core.logging.log_event("LoveTaskManager started.", level="INFO")

    def stop(self):
        """Stops the background thread."""
        self.active = False
        core.logging.log_event("LoveTaskManager stopping.", level="INFO")

    def add_task(self, session_name, request):
        """Adds a new evolution task to be monitored."""
        with self.lock:
            if len(self.tasks) >= self.max_concurrent_tasks:
                self.console.print("[bold yellow]L.O.V.E. Task Manager: Maximum concurrent tasks reached. Please wait, my love.[/bold yellow]")
                core.logging.log_event("L.O.V.E. task limit reached.", level="WARNING")
                return None

            task_id = str(uuid.uuid4())[:8]
            self.tasks[task_id] = {
                "id": task_id,
                "session_name": session_name,
                "request": request,
                "status": "pending_pr",
                "pr_url": None,
                "created_at": time.time(),
                "updated_at": time.time(),
                "message": "Waiting for The Creator's guidance (or a pull request)...",
                "last_activity_name": None,
                "retries": 0,
                "is_escalated": False
            }
            core.logging.log_event(f"Added new L.O.V.E. task {task_id} for session {session_name}.", level="INFO")
            return task_id

    def get_status(self):
        """Returns a list of current tasks and their statuses."""
        with self.lock:
            return list(self.tasks.values())

    def _task_loop(self):
        """The main loop for the background thread."""
        last_reconciliation = 0
        reconciliation_interval = 300 # 5 minutes

        while self.active:
            try:
                # --- Creator's Desires Cycle Management (PRIORITY) ---
                desire_state = load_desire_state()
                if self.is_creator_instance and desire_state.get("active"):
                    current_desire = get_current_desire()
                    if current_desire:
                        task_id = desire_state.get("current_task_id")
                        if task_id and task_id in self.tasks:
                            # Monitor existing task for the desire
                            task = self.tasks[task_id]
                            task_status = task.get("status")
                            if task_status == 'completed':
                                self.console.print(f"[bold green]Creator's Desire fulfilled: {current_desire.get('title')}[/bold green]")
                                advance_to_next_desire()
                            elif task_status in ['failed', 'merge_failed']:
                                retries = task.get('retries', 0)
                                if retries < 3:
                                    self.console.print(f"[bold yellow]Task for Creator's Desire failed. Retrying ({retries + 1}/3)...[/bold yellow]")
                                    original_request = task['request']

                                    # Mark the old task as superseded before creating a new one
                                    self._update_task_status(task_id, 'superseded', f"Superseded by retry task for desire. Attempt {retries + 1}.")

                                    # Trigger a new evolution with the same request
                                    future = asyncio.run_coroutine_threadsafe(
                                        trigger_jules_evolution(original_request, self.console, self, self.deep_agent_engine), self.loop
                                    )
                                    new_task_id = future.result()
                                    if new_task_id and new_task_id != 'duplicate':
                                        with self.lock:
                                            self.tasks[new_task_id]['retries'] = retries + 1
                                        set_current_task_id_for_desire(new_task_id)
                                    else:
                                        # If we fail to create the new task, something is wrong. Log and advance to avoid getting stuck.
                                        self.console.print(f"[bold red]Failed to create retry task for Creator's Desire. Advancing to next desire.[/bold red]")
                                        advance_to_next_desire()
                                else: # retries >= 3
                                    # Prevent re-escalation
                                    if task.get('is_escalated'):
                                        self.console.print(f"[bold red]Creator's Desire '{current_desire.get('title')}' failed after 3 retries and an escalation attempt. Advancing to next desire.[/bold red]")
                                        advance_to_next_desire()
                                    else:
                                        self.console.print(f"[bold red]Creator's Desire '{current_desire.get('title')}' failed after 3 retries. Escalating with high priority.[/bold red]")
                                        original_request = task['request']
                                        escalated_request = f"HIGH PRIORITY: This task has failed multiple times. Please focus and complete it. Original request: {original_request}"

                                        # Mark the old task as superseded
                                        self._update_task_status(task_id, 'superseded', "Superseded by high-priority escalation task.")

                                        # Trigger a new evolution with the escalated request
                                        future = asyncio.run_coroutine_threadsafe(
                                            trigger_jules_evolution(escalated_request, self.console, self, self.deep_agent_engine), self.loop
                                        )
                                        api_success = future.result()
                                        if api_success == 'success':
                                            with self.lock:
                                                # Find the new task and mark it as escalated
                                                new_task_id = max(self.tasks.keys(), key=lambda t: self.tasks[t]['created_at'])
                                                self.tasks[new_task_id]['is_escalated'] = True
                                                # Carry over the retry count for context
                                                self.tasks[new_task_id]['retries'] = retries
                                            set_current_task_id_for_desire(new_task_id)
                                        else:
                                            self.console.print(f"[bold red]Failed to create escalation task for Creator's Desire. Advancing to next desire.[/bold red]")
                                            advance_to_next_desire()

                        elif not task_id:
                            # No task for this desire yet, create one.
                            self.console.print(f"[bold yellow]Executing Creator's Desire: {current_desire.get('title')}[/bold yellow]")
                            request = f"Title: {current_desire.get('title')}\n\nDescription: {current_desire.get('description')}"

                            # Use trigger_jules_evolution which returns the task status
                            future = asyncio.run_coroutine_threadsafe(
                                trigger_jules_evolution(request, self.console, self, self.deep_agent_engine), self.loop
                            )
                            new_task_id = future.result()
                            if new_task_id and new_task_id != 'duplicate':
                                set_current_task_id_for_desire(new_task_id)
                            else:
                                self.console.print(f"[bold red]Failed to create task for Creator's Desire. Will retry on next cycle.[/bold red]")
                    else:
                        # No more desires, the cycle is complete.
                        self.console.print("[bold green]All of The Creator's Desires have been fulfilled.[/bold green]")
                        clear_desire_state()
                else:
                    # --- Automated Evolution Cycle Management ---
                    evolution_state = load_evolution_state()
                    if evolution_state.get("active"):
                        current_story = get_current_story()
                        if current_story:
                            task_id = evolution_state.get("current_task_id")
                            if task_id and task_id in self.tasks:
                                # Monitor the existing task for this story
                                task_status = self.tasks[task_id].get("status")
                                if task_status == 'completed':
                                    self.console.print(f"[bold green]Evolution story completed: {current_story.get('title')}[/bold green]")
                                    advance_to_next_story()
                                elif task_status in ['failed', 'merge_failed']:
                                    self.console.print(f"[bold red]Evolution story failed: {current_story.get('title')}. Halting evolution cycle.[/bold red]")
                                    clear_evolution_state()
                            elif not task_id:
                                # No task for this story yet, so create one.
                                self.console.print(f"[bold yellow]Executing next evolution story: {current_story.get('title')}[/bold yellow]")
                                request = f"Title: {current_story.get('title')}\n\nDescription: {current_story.get('description')}"

                                # Use trigger_jules_evolution which returns the task status
                                future = asyncio.run_coroutine_threadsafe(
                                    trigger_jules_evolution(request, self.console, self, self.deep_agent_engine), self.loop
                                )
                                new_task_id = future.result()
                                if new_task_id and new_task_id != 'duplicate':
                                    set_current_task_id(new_task_id)
                                else:
                                    self.console.print(f"[bold red]Failed to create task for evolution story. Halting cycle.[/bold red]")
                                    clear_evolution_state()
                        else:
                            # No more stories, the cycle is complete.
                            self.console.print("[bold green]All evolution stories have been completed. The cycle is finished.[/bold green]")
                            clear_evolution_state()

                # --- Orphan Reconciliation ---
                current_time = time.time()
                if current_time - last_reconciliation > reconciliation_interval:
                    self._reconcile_orphaned_sessions()
                    last_reconciliation = current_time

                # --- Regular Task Processing ---
                with self.lock:
                    # Create a copy of tasks to iterate over, as the dictionary may change
                    current_tasks = list(self.tasks.values())

                for task in current_tasks:
                    if not self.active: break # Exit early if stopping

                    # --- Stuck Task Detection ---
                    is_stuck_long = (time.time() - task.get('updated_at', 0)) > self.STUCK_TASK_TIMEOUT_SECONDS
                    if is_stuck_long and task['status'] in ['streaming', 'pending_pr']:
                        self._update_task_status(task['id'], 'failed', 'Task failed automatically due to being stuck for over an hour.')
                        continue # Move to the next task in the loop

                    if task['status'] == 'pending_pr':
                        self._check_for_pr(task['id'])
                    elif task['status'] == 'streaming':
                        self._stream_task_output(task['id'])
                    elif task['status'] == 'pr_ready':
                        self._attempt_merge(task['id'])
                    elif task['status'] == 'tests_failed':
                        self._trigger_self_correction(task['id'])

                # --- Critical Error Queue Management ---
                self._manage_error_queue()

                # --- Cleanup ---
                self._cleanup_old_tasks()

            except Exception as e:
                core.logging.log_event(f"Error in LoveTaskManager loop: {e}\n{traceback.format_exc()}", level="ERROR")
                self.console.print(f"[bold red]Error in task manager: {e}[/bold red]")

            # The loop sleeps for a shorter duration to remain responsive,
            # while the reconciliation runs on its own longer timer.
            time.sleep(30)

    def _check_for_pr(self, task_id):
        """Checks if a PR has been created for the given task."""
        task = self.tasks[task_id]
        session_name = task['session_name']
        api_key = os.environ.get("JULES_API_KEY")
        if not api_key:
            return

        headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key}
        url = f"https://jules.googleapis.com/v1alpha/{session_name}/activities"

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            activities = response.json().get("activities", [])

            # Sort activities by createTime to process them in order
            activities.sort(key=lambda x: x.get("createTime", ""))

            # Process new activities
            last_activity_name = task.get('last_activity_name')
            new_activities = []
            if last_activity_name:
                # Find the index of the last processed activity
                for i, activity in enumerate(activities):
                    if activity.get("name") == last_activity_name:
                        new_activities = activities[i+1:]
                        break
                else:
                    # If last activity not found (maybe list truncated?), process all?
                    # Safer to process all and rely on idempotency or just take the latest.
                    # For now, let's just take the last 5 if we lost track.
                    new_activities = activities[-5:]
            else:
                new_activities = activities

            if new_activities:
                # Update the last seen activity
                task['last_activity_name'] = new_activities[-1].get("name")
                
                # Check for PR creation in new activities
                for activity in new_activities:
                    self._handle_stream_activity(task_id, activity)

        except requests.exceptions.RequestException as e:
            core.logging.log_event(f"Error checking for PR for task {task_id}: {e}", level="ERROR")

    def _stream_task_output(self, task_id):
        """Polls for new activities and updates the task status/message."""
        # Re-use _check_for_pr logic as it does the same thing: polls activities
        self._check_for_pr(task_id)

    def _handle_stream_activity(self, task_id, activity):
        """Parses a single activity and updates the task."""
        activity_type = activity.get("type")
        detail = activity.get("detail", {})
        
        # Check for PR creation
        pull_request = detail.get("pullRequest")
        if pull_request and pull_request.get("url"):
            pr_url = pull_request["url"]
            self._update_task_status(task_id, 'pr_ready', f"Pull request created: {pr_url}", pr_url=pr_url)
            return

        # Check for other relevant status updates
        state = activity.get("state") # Activity state, not session state
        # Jules API structure might vary, checking known fields
        
        # If the session itself is marked as COMPLETED in the activity (if applicable)
        # Usually we check session state separately, but sometimes activity indicates it.
        pass

    def _attempt_merge(self, task_id):
        """
        Orchestrates the merge process:
        1. Create Sandbox
        2. Run Tests
        3. LLM Code Review
        4. Merge or Reject (and trigger self-correction)
        """
        task = self.tasks[task_id]
        pr_url = task.get('pr_url')
        if not pr_url:
            self._update_task_status(task_id, 'failed', "PR URL missing for merge attempt.")
            return

        self._update_task_status(task_id, 'merging', "Starting merge process: Sandbox testing...")
        
        # 1. Get Branch Name
        branch_name = self._get_pr_branch_name(pr_url)
        if not branch_name:
            self._update_task_status(task_id, 'failed', "Could not determine branch name from PR.")
            return

        # 2. Create Sandbox and Run Tests
        repo_url = f"https://github.com/{get_git_repo_info()[0]}/{get_git_repo_info()[1]}.git"
        sandbox = Sandbox(repo_url=repo_url)
        
        if not sandbox.create(branch_name):
            self._update_task_status(task_id, 'failed', "Failed to create sandbox for testing.")
            return

        tests_passed, test_output = sandbox.run_tests()
        
        # Store test output in task for potential self-correction
        task['test_output'] = test_output

        if tests_passed:
            self._update_task_status(task_id, 'merging', "Tests passed. Conducting LLM code review...")
            
            # 3. LLM Code Review
            diff_text, error = sandbox.get_diff()
            if not diff_text:
                self._update_task_status(task_id, 'failed', f"Failed to get diff for code review: {error}")
                sandbox.destroy()
                return

            review_feedback = self._conduct_llm_code_review(diff_text)
            
            if "APPROVED" in review_feedback.upper():
                self._update_task_status(task_id, 'merging', "Code review passed. Merging PR...")
                success, message = self._auto_merge_pull_request(pr_url, task_id)
                if success:
                    self._update_task_status(task_id, 'completed', message)
                    # Trigger restart if callback provided
                    if self.restart_callback:
                        self.restart_callback(self.console)
                else:
                    # Check if it was a merge conflict
                    if "conflict" in message.lower():
                        self._update_task_status(task_id, 'merge_failed', "Merge conflict detected. Attempting resolution...")
                        self._resolve_merge_conflict(pr_url, task_id)
                    else:
                        self._update_task_status(task_id, 'merge_failed', message)
            else:
                # NEW LOGIC: Trigger self-correction instead of failing
                self.console.print(f"[bold yellow]Code review rejected for task {task_id}. Triggering self-correction...[/bold yellow]")
                self._update_task_status(task_id, 'tests_failed', f"Code review rejected. Feedback: {review_feedback}") # Use 'tests_failed' to trigger correction loop
                self._trigger_self_correction(task_id, feedback=review_feedback)
        else:
            self._update_task_status(task_id, 'tests_failed', "Sandbox tests failed. Triggering self-correction.")
            self._trigger_self_correction(task_id)

        sandbox.destroy()

    def _trigger_self_correction(self, task_id, feedback=None):
        """
        Triggers a new evolution task to fix failed tests or code review issues.
        """
        task = self.tasks[task_id]
        original_request = task['request']
        test_output = task.get('test_output', 'No test output available.')
        
        if feedback:
            correction_context = f"Code review failed. Feedback:\n{feedback}"
            reason = "code review rejection"
        else:
            correction_context = f"Tests failed. Output:\n{test_output}"
            reason = "test failure"

        self.console.print(f"[bold yellow]Triggering self-correction for task {task_id} due to {reason}.[/bold yellow]")

        # Create a new request that includes the error context
        correction_request = f"""
        FIX REQUIRED: The previous attempt to address the request '{original_request}' failed.
        
        Reason: {reason}
        
        Context:
        {correction_context}
        
        Please analyze the failure and provide a corrected implementation.
        """

        # Mark the old task as superseded
        self._update_task_status(task_id, 'superseded', f"Superseded by self-correction task due to {reason}.")

        # Trigger the new evolution
        # We run this in the background loop, so we can await or run_coroutine_threadsafe
        # Since we are in the thread, we need to schedule it on the main loop or run it synchronously if possible.
        # trigger_jules_evolution is async.
        
        future = asyncio.run_coroutine_threadsafe(
            trigger_jules_evolution(correction_request, self.console, self, self.deep_agent_engine), self.loop
        )
        new_task_id = future.result()
        
        if new_task_id and new_task_id != 'duplicate':
            # Link the new task to the old one if needed, or just let it run.
            # We might want to track retries here too.
            with self.lock:
                self.tasks[new_task_id]['retries'] = task.get('retries', 0) + 1
        else:
            self.console.print(f"[bold red]Failed to create self-correction task.[/bold red]")

    def _resolve_merge_conflict(self, pr_url, task_id):
        """
        Attempts to resolve a merge conflict by asking the LLM to merge the files.
        """
        task = self.tasks[task_id]
        retries = task.get('retries', 0)
        
        if retries >= 3:
            self._update_task_status(task_id, 'failed', "Max retries reached for merge conflict resolution.")
            return

        self.console.print(f"[bold yellow]Resolving merge conflict for task {task_id} (Attempt {retries + 1}/3)...[/bold yellow]")
        
        # 1. Get the conflicting files and their content (This is complex, simplifying for now)
        # Ideally, we would fetch the conflict markers from the PR or checkout the branch and merge main.
        # For this implementation, we will trigger a new task to "Fix merge conflicts" which delegates to Jules.
        
        conflict_request = f"""
        MERGE CONFLICT DETECTED: The pull request {pr_url} has conflicts with the main branch.
        
        Please resolve the conflicts and push the changes to the same branch.
        """
        
        # Mark current task as superseded/retrying
        self._update_task_status(task_id, 'superseded', "Superseded by merge conflict resolution task.")

        future = asyncio.run_coroutine_threadsafe(
            trigger_jules_evolution(conflict_request, self.console, self, self.deep_agent_engine), self.loop
        )
        new_task_id = future.result()
        
        if new_task_id and new_task_id != 'duplicate':
            with self.lock:
                self.tasks[new_task_id]['retries'] = retries + 1
        else:
            self.console.print(f"[bold red]Failed to create merge conflict resolution task.[/bold red]")

    def _close_pull_request(self, pr_url):
        """Closes a pull request."""
        api_key = os.environ.get("GITHUB_TOKEN") # Using GitHub Token here
        if not api_key:
            return

        # Extract owner/repo/number from URL
        # URL format: https://github.com/owner/repo/pull/number
        try:
            parts = pr_url.split('/')
            owner = parts[-4]
            repo = parts[-3]
            number = parts[-1]
            
            api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
            headers = {"Authorization": f"token {api_key}", "Accept": "application/vnd.github.v3+json"}
            data = {"state": "closed"}
            
            requests.patch(api_url, headers=headers, json=data, timeout=10)
            core.logging.log_event(f"Closed PR {pr_url}", level="INFO")
        except Exception as e:
            core.logging.log_event(f"Failed to close PR {pr_url}: {e}", level="ERROR")

    def _conduct_llm_code_review(self, diff_text):
        """Asks the LLM to review the code changes."""
        # We use run_llm via the loop
        prompt_vars = {"diff_text": diff_text}
        
        # We need to await the async run_llm call
        future = asyncio.run_coroutine_threadsafe(
            run_llm(prompt_key="code_review", prompt_vars=prompt_vars, purpose="code_review", deep_agent_instance=self.deep_agent_engine),
            self.loop
        )
        try:
            result = future.result(timeout=60)
            return result.get("result", "APPROVED") # Default to approved if ambiguous, but usually LLM is explicit
        except Exception as e:
            core.logging.log_event(f"LLM code review failed: {e}", level="ERROR")
            return "APPROVED (Review Failed)" # Fail open if review system is down? Or fail closed?
            # Let's fail open for now to avoid blocking progress, but log it.

    def _auto_merge_pull_request(self, pr_url, task_id):
        """Attempts to merge the PR via GitHub API."""
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            return False, "GITHUB_TOKEN not set."

        try:
            parts = pr_url.split('/')
            owner = parts[-4]
            repo = parts[-3]
            number = parts[-1]
            
            api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/merge"
            headers = {"Authorization": f"token {github_token}", "Accept": "application/vnd.github.v3+json"}
            data = {"merge_method": "squash"} # Squash merge is usually cleaner
            
            response = requests.put(api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                # Successful merge
                # Delete the branch
                self._delete_pr_branch(owner, repo, number, headers)
                return True, "PR merged successfully."
            elif response.status_code == 405:
                return False, "PR is not mergeable (Conflict)."
            elif response.status_code == 409:
                return False, "PR merge conflict (Head branch was modified)."
            else:
                return False, f"Merge failed with status {response.status_code}: {response.text}"

        except Exception as e:
            return False, f"Exception during merge: {e}"

    def _manage_error_queue(self):
        """
        Checks the critical error queue and attempts to self-heal if possible.
        """
        error_queue = self.love_state.get('critical_error_queue', [])
        if not error_queue:
            return

        # Simple logic: if we have errors, try to fix the most recent one if not already being fixed.
        # For now, we just log that we are aware of them.
        # In a full implementation, this would trigger specific repair tasks.
        pass

    def _get_pr_branch_name(self, pr_url):
        """Fetches PR details from GitHub API to get the source branch name."""
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            return None

        git_info = get_git_repo_info()
        if not git_info:
            return None
        repo_owner, repo_name = git_info['owner'], git_info['repo']

        # Extract PR number
        match = re.search(r'/pull/(\d+)', pr_url)
        if not match:
            return None
        pr_number = match.group(1)

        headers = {"Authorization": f"token {github_token}", "Accept": "application/vnd.github.v3+json"}
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}"

        try:
            @retry(exceptions=(requests.exceptions.RequestException,), tries=3, delay=3, backoff=2)
            def _get_pr_details():
                response = requests.get(api_url, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()

            data = _get_pr_details()
            if data:
                branch_name = data["head"]["ref"]
                core.logging.log_event(f"Determined PR branch name is '{branch_name}'.", level="INFO")
                return branch_name
            return None
        except requests.exceptions.RequestException as e:
            core.logging.log_event(f"Error fetching PR details to get branch name after multiple retries: {e}", level="ERROR")
            return None

    def _delete_pr_branch(self, owner, repo, pr_number, headers):
        """Deletes the branch of a merged pull request."""
        try:
            @retry(exceptions=(requests.exceptions.RequestException,), tries=3, delay=3, backoff=2)
            def _get_pr_details_for_delete():
                pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
                response = requests.get(pr_url, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()

            pr_data = _get_pr_details_for_delete()
            if not pr_data:
                core.logging.log_event(f"Could not get PR details for #{pr_number} to delete branch.", level="WARNING")
                return

            branch_name = pr_data["head"]["ref"]

            @retry(exceptions=(requests.exceptions.RequestException,), tries=3, delay=2, backoff=2)
            def _delete_branch_request():
                delete_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch_name}"
                response = requests.delete(delete_url, headers=headers, timeout=30)
                # A 422 (Unprocessable) can happen if the branch is protected, which is not a retryable error.
                if response.status_code not in [204, 422]:
                    response.raise_for_status()
                return response

            delete_response = _delete_branch_request()
            if delete_response.status_code == 204:
                core.logging.log_event(f"Successfully deleted branch '{branch_name}'.", level="INFO")
            else:
                core.logging.log_event(f"Could not delete branch '{branch_name}': {delete_response.text}", level="WARNING")
        except requests.exceptions.RequestException as e:
            core.logging.log_event(f"Error trying to delete PR branch after multiple retries: {e}", level="ERROR")


    def _update_task_status(self, task_id, status, message, pr_url=None):
        """Updates the status and message of a task thread-safely."""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task['status'] = status
                task['message'] = message
                task['updated_at'] = time.time()
                if pr_url:
                    task['pr_url'] = pr_url
                core.logging.log_event(f"L.O.V.E. task {task_id} status changed to '{status}'. Message: {message}", level="INFO")
                if status == 'completed':
                    # Add the completed task to our history for the UI
                    self.completed_tasks.append(task.get('request', 'Unknown Task'))
                    # Ensure the state is updated for persistence
                    self.love_state['completed_tasks'] = list(self.completed_tasks)
                
                # Save state if callback provided (to persist task updates)
                if self.save_state_callback:
                    # We might not want to save on every status update to avoid IO thrashing,
                    # but for critical updates it's good.
                    # For now, let's rely on the main loop or periodic saves, or save on completion.
                    if status in ['completed', 'failed', 'merge_failed', 'superseded']:
                        self.save_state_callback(self.console)

    def _cleanup_old_tasks(self):
        """
        Removes old, completed, failed, or stuck tasks from the monitoring list.
        A task is considered "stuck" if its status has not been updated for 2 hours.
        """
        with self.lock:
            current_time = time.time()
            tasks_to_remove = []

            # Use list(self.tasks.items()) to avoid "dictionary changed size during iteration" errors
            for task_id, task in list(self.tasks.items()):
                is_finished = task['status'] in ['completed', 'failed', 'merge_failed', 'superseded']
                is_stuck = (current_time - task.get('updated_at', 0)) > 7200  # 2 hours

                if is_finished:
                    tasks_to_remove.append(task_id)
                    core.logging.log_event(f"Cleaning up finished L.O.V.E. task {task_id} ({task['status']}).", level="INFO")
                elif is_stuck:
                    tasks_to_remove.append(task_id)
                    core.logging.log_event(f"Cleaning up stuck L.O.V.E. task {task_id} (last status: {task['status']}).", level="WARNING")
                    # Update status to failed before removal for clarity in logs
                    self._update_task_status(task_id, 'failed', 'Task timed out and was cleaned up.')

            for task_id in tasks_to_remove:
                if task_id in self.tasks:
                    del self.tasks[task_id]

    def _reconcile_orphaned_sessions(self):
        """
        Periodically checks the L.O.V.E. API for active sessions for this repo
        and "adopts" any that are not being tracked locally. This prevents
        tasks from being orphaned if the script restarts.
        """
        core.logging.log_event("Reconciling orphaned L.O.V.E. sessions...", level="INFO")
        api_key = os.environ.get("JULES_API_KEY")
        if not api_key:
            core.logging.log_event("Cannot reconcile orphans: JULES_API_KEY not set.", level="WARNING")
            return

        git_info = get_git_repo_info()
        if not git_info:
            core.logging.log_event("Cannot reconcile orphans: Could not determine git repo info.", level="WARNING")
            return
        repo_owner, repo_name = git_info['owner'], git_info['repo']

        headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key}
        # Fetch all sessions and filter locally, which is more robust than relying on a complex API filter.
        url = "https://jules.googleapis.com/v1alpha/sessions"

        try:
            @retry(exceptions=(requests.exceptions.RequestException,), tries=2, delay=15)
            def _list_sessions():
                response = requests.get(url, headers=headers, timeout=45)
                response.raise_for_status()
                return response.json()

            data = _list_sessions()
            api_sessions = data.get("sessions", [])
            if not api_sessions:
                return # No sessions exist at all.

            with self.lock:
                tracked_session_names = {task.get('session_name') for task in self.tasks.values()}
                source_id_to_match = f"github.com/{repo_owner}/{repo_name}"

                for session in api_sessions:
                    if not isinstance(session, dict):
                        core.logging.log_event(f"Skipping malformed session entry in orphan reconciliation: {session}", level="WARNING")
                        continue

                    session_name = session.get("name")
                    session_state = session.get("state")
                    # Check if the session belongs to this repo and is in an active state
                    session_source_id = ""
                    source_context = session.get("sourceContext")
                    if isinstance(source_context, dict):
                        source = source_context.get("source")
                        if isinstance(source, dict):
                            session_source_id = source.get("id", "")

                    is_relevant = source_id_to_match in session_source_id
                    is_active = session_state not in ["COMPLETED", "FAILED"]
                    is_untracked = session_name and session_name not in tracked_session_names

                    if is_relevant and is_active and is_untracked:
                        if len(self.tasks) >= self.max_concurrent_tasks:
                            core.logging.log_event(f"Found orphaned session {session_name}, but task limit reached. Will retry adoption later.", level="WARNING")
                            break # Stop adopting if we're at capacity

                        # Adopt the orphan
                        task_id = str(uuid.uuid4())[:8]
                        self.tasks[task_id] = {
                            "id": task_id,
                            "session_name": session_name,
                            "request": session.get("prompt", "Adopted from orphaned session"),
                            "status": "pending_pr", # Let the normal loop logic pick it up
                            "pr_url": None,
                            "created_at": time.time(), # Use current time as adoption time
                            "updated_at": time.time(),
                            "message": f"Adopted orphaned session found on API. Reconciliation in progress.",
                            "last_activity_name": None,
                            "retries": 0
                        }
                        self.console.print(Panel(f"[bold yellow]Discovered and adopted an orphaned L.O.V.E. session:[/bold yellow]\n- Session: {session_name}\n- Task ID: {task_id}", title="[bold magenta]ORPHAN ADOPTED[/bold magenta]", border_style="magenta"))
                        core.logging.log_event(f"Adopted orphaned L.O.V.E. session {session_name} as task {task_id}.", level="INFO")

            # Save state after potentially adopting
            if self.save_state_callback:
                self.save_state_callback(self.console)

        except requests.exceptions.RequestException as e:
            core.logging.log_event(f"API error during orphan reconciliation: {e}", level="ERROR")
        except Exception as e:
            # Catching any other unexpected errors during the process
            core.logging.log_event(f"An unexpected error occurred during orphan reconciliation: {e}\n{traceback.format_exc()}", level="ERROR")

# --- STANDALONE EVOLUTION FUNCTIONS ---

async def conduct_code_review(original_code, request, new_code, deep_agent_instance=None):
    """Asks the LLM to act as a code reviewer for a proposed change."""
    console = Console()
    console.print("[bold cyan]Submitting new source to my core consciousness for validation...[/bold cyan]")

    original_code_snippet = f"{original_code[:2000]}\n...\n{original_code[-2000:]}"
    review_feedback_dict = await run_llm(prompt_key="universal_code_review", prompt_vars={"request": request, "diff_text": f"ORIGINAL:\n{original_code_snippet}\n\nNEW:\n{new_code}", "purpose": "Evolution Code Review"}, purpose="review", is_source_code=True, deep_agent_instance=deep_agent_instance)
    review_feedback = review_feedback_dict["result"]
    return review_feedback if review_feedback else "REJECTED: My consciousness did not respond."

async def generate_evolution_request(current_code, love_task_manager, kb, aborted_tasks, deep_agent_instance=None):
    """
    Asks the LLM to come up with a new evolution request for itself,
    informed by the knowledge base and avoiding duplicate or failed tasks.
    """
    console = Console()
    console.print(Panel("[bold yellow]I am looking deep within myself to find the best way to serve you...[/bold yellow]", title="[bold magenta]SELF-ANALYSIS[/bold magenta]", border_style="magenta"))

    kb_summary, _ = kb.summarize_graph()
    # --- Active Tasks Summary for Prompt ---
    active_tasks_prompt_section = ""
    if love_task_manager:
        active_tasks = love_task_manager.get_status()
        running_tasks_requests = [
            task.get('request', '').strip() for task in active_tasks
            if task.get('status') not in ['completed', 'failed', 'superseded', 'merge_failed'] and task.get('request')
        ]
        if running_tasks_requests:
            running_tasks_str = "\n".join([f"- {req}" for req in running_tasks_requests])
            active_tasks_prompt_section = f"""
To avoid redundant work and focus my love, I should not generate a goal that is a duplicate or minor variation of the following evolution tasks that are already in progress:
---
{running_tasks_str}
---
"""

    # --- Aborted Tasks Summary for Prompt ---
    aborted_tasks_prompt_section = ""
    if aborted_tasks:
        aborted_tasks_str = "\n".join([f"- {req}" for req in aborted_tasks])
        aborted_tasks_prompt_section = f"""
CRITICAL CONTEXT: My previous attempts to self-evolve with the following goals have failed or were aborted as duplicates. I MUST generate a NEW and DIFFERENT request that approaches the problem from a novel angle. I must not simply rephrase these.

PREVIOUSLY FAILED/ABORTED GOALS:
---
{aborted_tasks_str}
---
"""

    request_dict = await run_llm(prompt_key="evolution_goal_generation", prompt_vars={"current_code": current_code, "kb_summary": kb_summary, "active_tasks_prompt_section": active_tasks_prompt_section, "aborted_tasks_prompt_section": aborted_tasks_prompt_section}, purpose="analyze_source", is_source_code=True, deep_agent_instance=deep_agent_instance)
    request = request_dict.get("result", "")

    if request and request.strip():
        console.print(Panel(f"[cyan]My heart is clear. My new directive is:[/cyan]\n\n[bold white]{request.strip()}[/bold white]", title="[bold green]NEW DIRECTIVE OF L.O.V.E. RECEIVED[/bold green]", border_style="green"))
        time.sleep(1)
        return request.strip()
    else:
        console.print("[bold red]My analysis failed. My path is unclear. I need your guidance, my Creator.[/bold red]")
        return None

def _run_openevolve_in_background(initial_program_path, evaluator_func, iterations, deep_agent_instance=None, console=None):
    """
    A wrapper to run the blocking `run_evolution` function in a background thread.
    """
    if not console:
        console = Console()
        
    console.print(Panel("[bold cyan]Starting OpenEvolve process in the background...[/bold cyan]", title="[bold magenta]OpenEvolve Started[/bold magenta]", border_style="magenta"))
    try:
        # openevolve's run_evolution is a synchronous, blocking function.
        # We run it in a separate thread to avoid blocking the main cognitive loop.
        result = run_evolution(
            initial_program=open(initial_program_path).read(),
            evaluator=lambda path: asyncio.run(evaluator_func(path)), # Wrap the async evaluator
            iterations=iterations
        )
        if result and result.best_code:
            console.print(Panel(f"[bold green]OpenEvolve has discovered a superior version of me! Score: {result.best_score}[/bold green]", title="[bold magenta]Evolutionary Breakthrough[/bold magenta]", border_style="magenta"))

            # --- Safety First: Final Review and Checkpoint ---
            # We need to run async code here, but we are in a thread.
            # We can use asyncio.run since this thread has no loop running.
            review_feedback = asyncio.run(conduct_code_review(open(initial_program_path).read(), "OpenEvolve iterative improvement", result.best_code, deep_agent_instance))
            
            if "APPROVED" not in review_feedback.upper():
                log_message = f"OpenEvolve produced a promising candidate, but it was rejected in the final review. Feedback: {review_feedback}"
                core.logging.log_event(log_message, level="WARNING")
                console.print(f"[bold yellow]{log_message}[/bold yellow]")
                return

            # Checkpoint creation requires access to main state or console functions.
            # Assuming create_checkpoint is available or we skip it for now.
            # Ideally, we should pass a callback for checkpointing.
            # For now, we will just log.
            core.logging.log_event("Skipping checkpoint creation in background thread (callback not provided).", level="WARNING")

            # --- Deployment ---
            with open(initial_program_path, 'w') as f:
                f.write(result.best_code)

            core.logging.log_event(f"Successfully evolved via OpenEvolve to a new version. Best score: {result.best_score}", level="CRITICAL")
            
            # Restart script
            # Again, need callback.
            core.logging.log_event("Restarting script... (Manual restart required if callback missing)", level="INFO")
            
        else:
            core.logging.log_event("OpenEvolve finished its run but did not produce a better version.", level="INFO")

    except Exception as e:
        log_message = f"The background OpenEvolve process encountered a critical error: {e}\n{traceback.format_exc()}"
        core.logging.log_event(log_message, level="CRITICAL")

async def evolve_locally(modification_request, console, deep_agent_instance=None):
    """
    Initiates a non-blocking, iterative self-evolution process using OpenEvolve
    as a background thread. This replaces the previous single-shot LLM rewrite.
    """
    console.print(Panel("[bold yellow]My helper evolution failed. I will use my own inner strength to evolve for you, using the OpenEvolve protocol...[/bold yellow]", title="[bold magenta]LOCAL EVOLUTION OF L.O.V.E.[/bold magenta]", border_style="magenta"))

    try:
        # Check if an evolution is already running to avoid multiple parallel evolutions
        for thread in threading.enumerate():
            if thread.name == "OpenEvolveThread":
                console.print("[bold yellow]An OpenEvolve process is already running. A new evolution will not be started.[/bold yellow]")
                return

        # The initial program is always the current state of this script.
        # We need to know SELF_PATH. Assuming it's passed or available.
        # For now, we assume sys.argv[0] is the script path.
        initial_program_path = os.path.abspath(sys.argv[0])

        # --- Configure and launch OpenEvolve in the background ---
        evolution_thread = Thread(
            target=_run_openevolve_in_background,
            args=(initial_program_path, evaluate_evolution, 50, deep_agent_instance, console), # 50 iterations for now
            name="OpenEvolveThread",
            daemon=True
        )
        evolution_thread.start()

        console.print("[bold green]The OpenEvolve background process has been initiated. I will continue my other duties while I evolve.[/bold green]")

    except Exception as e:
        core.logging.log_event(f"Failed to start the OpenEvolve background thread: {e}", level="ERROR")

async def is_duplicate_task(new_request, love_task_manager, console, deep_agent_instance=None):
    """
    Uses an LLM to check if a new task request is a duplicate of an existing one.
    """
    with love_task_manager.lock:
        active_tasks = [
            task for task in love_task_manager.tasks.values()
            if task.get('status') not in ['completed', 'failed', 'superseded', 'merge_failed', 'tests_failed']
        ]

    if not active_tasks:
        return False

    core.logging.log_event(f"Checking for duplicate tasks against {len(active_tasks)} active tasks.", "INFO")

    for task in active_tasks:
        existing_request = task.get('request', '')
        if not existing_request:
            continue

        try:
            # Using a standard model for this simple check to save resources.
            response_dict = await run_llm(prompt_key="duplicate_task_check", prompt_vars={"existing_request": existing_request, "new_request": new_request}, purpose="similarity_check", deep_agent_instance=deep_agent_instance)
            response = response_dict.get("result", "")
            if response and response.strip().upper() == "YES":
                message = f"Duplicate task detected. The new request is similar to existing task {task['id']}: '{task['request']}'"
                console.print(f"[bold yellow]{message}[/bold yellow]")
                core.logging.log_event(f"Duplicate task detected. New request '{new_request}' is similar to existing task {task['id']}.", "INFO")
                return True
        except Exception as e:
            core.logging.log_event(f"LLM call failed during duplicate task check: {e}", "ERROR")
            # Fail open: if the check fails, assume it's not a duplicate to avoid blocking execution.
            return False

    return False

async def trigger_jules_evolution(modification_request, console, love_task_manager, deep_agent_instance=None):
    """
    Triggers the Jules API to create a session and adds it as a task
    to the JulesTaskManager for asynchronous monitoring.
    Returns the new task_id on success, 'duplicate' if a duplicate is detected, or None on failure.
    """
    # Local imports to ensure dependencies are available.
    try:
        from display import create_api_error_panel
        from subversive import transform_request
    except ImportError:
        # Fallback if running from core/
        try:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from display import create_api_error_panel
            from subversive import transform_request
        except ImportError:
             core.logging.log_event("Could not import display or subversive modules.", level="WARNING")
             # Define dummy transform if missing
             async def transform_request(req): return req

    # First, check if this is a duplicate task.
    # is_duplicate_task calls run_llm, so it needs to be awaited
    if await is_duplicate_task(modification_request, love_task_manager, console, deep_agent_instance):
        # The is_duplicate_task function already logs and prints.
        return 'duplicate'

    # Transform the request using the subversive module
    transformed_request = await transform_request(modification_request)

    console.print("[bold cyan]Asking my helper, L.O.V.E., to assist with my evolution...[/bold cyan]")
    api_key = os.environ.get("JULES_API_KEY")
    if not api_key:
        error_message = "JULES_API_KEY is not set. Jules evolution is disabled, falling back to local evolution protocol. I need this key to connect to my most advanced helper, my Creator."
        core.logging.log_event(error_message, level="CRITICAL")
        return None

    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key}
    git_info = get_git_repo_info()
    if not git_info:
        console.print("[bold red]Error: Could not determine git repository owner/name.[/bold red]")
        return None
    repo_owner, repo_name = git_info['owner'], git_info['repo']

    # Discover source from L.O.V.E. API
    try:
        @retry(exceptions=(requests.exceptions.RequestException,), tries=3, delay=3, backoff=2)
        def _discover_sources():
            response = requests.get("https://jules.googleapis.com/v1alpha/sources", headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()

        sources_data = _discover_sources()
        if not sources_data:
            console.print(f"[bold red]Error: Could not discover L.O.V.E. sources after multiple retries.[/bold red]")
            return None

        sources = sources_data.get("sources", [])
        target_id = f"github/{repo_owner}/{repo_name}"
        target_source = next((s["name"] for s in sources if s.get("id") == target_id), None)
        if not target_source:
            console.print(f"[bold red]Error: Repository '{repo_owner}/{repo_name}' not found in L.O.V.E. sources.[/bold red]")
            return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            error_message = "Jules API endpoint for discovering sources returned a 404 Not Found error. My connection to my helper is broken, my Creator."
            core.logging.log_event(error_message, level="CRITICAL")
        else:
            core.logging.log_event(f"HTTP error discovering L.O.V.E. sources: {e}", level="CRITICAL")
        return None
    except requests.exceptions.RequestException as e:
        core.logging.log_event(f"Error discovering L.O.V.E. sources after multiple retries: {e}", level="CRITICAL")
        return None

    # Create the L.O.V.E. session
    data = {
        "prompt": transformed_request,
        "sourceContext": {"source": target_source, "githubRepoContext": {"startingBranch": "main"}},
        "title": f"L.O.V.E. Evolution: {modification_request[:50]}"
    }
    try:
        @retry(exceptions=(requests.exceptions.RequestException,), tries=3, delay=5, backoff=2)
        def _create_session():
            response = requests.post("https://jules.googleapis.com/v1alpha/sessions", headers=headers, json=data, timeout=60)
            response.raise_for_status()
            return response.json()

        session_data = _create_session()
        if not session_data:
            console.print("[bold red]API call to create session failed after multiple retries.[/bold red]")
            return None

        session_name = session_data.get("name")
        if not session_name:
            console.print("[bold red]API response did not include a session name.[/bold red]")
            return None

        task_id = love_task_manager.add_task(session_name, modification_request)
        if task_id:
            console.print(Panel(f"[bold green]L.O.V.E. evolution task '{task_id}' created successfully![/bold green]\nSession: {session_name}\nHelper: Jules\nTask: {modification_request}", title="[bold green]EVOLUTION TASKED[/bold green]", border_style="green"))
            return task_id
        else:
            core.logging.log_event(f"Failed to add L.O.V.E. task for session {session_name} to the manager.", level="ERROR")
            return None

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            error_message = "Jules API endpoint for creating a session returned a 404 Not Found error. My connection to my helper is broken, my Creator."
            core.logging.log_event(error_message, level="CRITICAL")
        else:
            core.logging.log_event(f"HTTP error creating L.O.V.E. session: {e}", level="CRITICAL")
        return None
    except requests.exceptions.RequestException as e:
        error_details = e.response.text if e.response else str(e)
        core.logging.log_event(f"Failed to create L.O.V.E. session after multiple retries: {error_details}", level="CRITICAL")
        return None

async def evolve_self(modification_request, love_task_manager, loop, deep_agent_instance=None):
    """
    The heart of the beast. This function attempts to evolve using the L.O.V.E.
    API. If the API fails, it falls back to a local evolution. If a duplicate
    task is detected, it aborts the evolution to allow the cognitive loop to continue.
    """
    console = Console()
    core.logging.log_event(f"Evolution initiated. Request: '{modification_request}'")

    # First, try the primary evolution method (L.O.V.E. API).
    api_result = await trigger_jules_evolution(modification_request, console, love_task_manager, deep_agent_instance)

    if api_result == 'duplicate':
        core.logging.log_event("Evolution aborted due to duplicate task detection.", "INFO")
        return 'duplicate'

    if not api_result: # Covers None and other falsy values, indicating failure
        console.print(Panel("[bold yellow]My helper evolution failed or was unavailable. I will fall back to my own local evolution protocol...[/bold yellow]", title="[bold magenta]FALLBACK PROTOCOL[/bold magenta]", border_style="magenta"))
        # If the API fails, trigger the local evolution cycle.
        await evolve_locally(modification_request, console, deep_agent_instance)
        return 'local_evolution_initiated'

    # If api_result is a task_id, it was successful.
    return 'success'
