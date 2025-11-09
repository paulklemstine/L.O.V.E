import pytest
import os
import time
import uuid
import subprocess
import re
import requests

# Ensure the app's root directory is in the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from love import JulesTaskManager, trigger_jules_evolution, get_git_repo_info
from rich.console import Console
from core.retry import retry

# --- Configuration ---
# Ensure JULES_API_KEY and GITHUB_TOKEN are set as environment variables
TEST_TIMEOUT = 900  # 15 minutes
POLL_INTERVAL = 30  # 30 seconds

def _get_pr_branch_name(pr_url):
    """Fetches PR details from GitHub API to get the source branch name."""
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        return None

    repo_owner, repo_name = get_git_repo_info()
    if not repo_owner or not repo_name:
        return None

    pr_number_match = re.search(r'/pull/(\d+)', pr_url)
    if not pr_number_match:
        return None
    pr_number = pr_number_match.group(1)

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
            return data["head"]["ref"]
        return None
    except requests.exceptions.RequestException:
        return None

def _delete_pr_branch(owner, repo, pr_number, headers):
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
            return

        branch_name = pr_data["head"]["ref"]

        @retry(exceptions=(requests.exceptions.RequestException,), tries=3, delay=2, backoff=2)
        def _delete_branch_request():
            delete_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch_name}"
            response = requests.delete(delete_url, headers=headers, timeout=30)
            if response.status_code not in [204, 422]:
                response.raise_for_status()
            return response

        _delete_branch_request()
    except requests.exceptions.RequestException:
        pass


@pytest.fixture(scope="module")
def console():
    """Provides a Rich Console instance for the test module."""
    return Console()

@pytest.fixture(scope="module")
def task_manager(console):
    """
    Provides a live JulesTaskManager instance. The background loop is allowed
    to run as it would in production.
    """
    manager = JulesTaskManager(console=console)
    manager.start()
    yield manager
    manager.stop()

# Skip this test if the required API keys are not set
JULES_API_KEY = os.environ.get("JULES_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
pytestmark = pytest.mark.skipif(not JULES_API_KEY or not GITHUB_TOKEN, reason="JULES_API_KEY and GITHUB_TOKEN environment variables not set")


@pytest.mark.real_api
@pytest.mark.asyncio
async def test_jules_increment_counter_lifecycle(task_manager, console):
    """
    Tests the full lifecycle of the Jules/L.O.V.E. system by incrementing a counter file.
    1. Reads the initial value from 'jules_build_counter.txt'.
    2. Triggers a Jules task to increment the value in the file.
    3. Monitors the task until the PR is successfully merged.
    4. Verifies that the file on the main branch has been updated with the incremented value.
    """
    counter_file = "jules_build_counter.txt"
    repo_owner, repo_name = get_git_repo_info()
    assert repo_owner and repo_name, "Could not determine repository info."

    # --- 1. Read Initial State ---
    subprocess.check_call(["git", "checkout", "main"])
    subprocess.check_call(["git", "pull", "origin", "main"])

    assert os.path.exists(counter_file), f"Counter file '{counter_file}' not found."
    with open(counter_file, "r") as f:
        initial_value = int(f.read().strip())

    console.print(f"[bold cyan]Initial counter value is: {initial_value}[/bold cyan]")

    # --- 2. Trigger the Evolution Task ---
    request = f"Read the integer from `{counter_file}`, increment it by one, and write the new value back to the file."

    success = trigger_jules_evolution(request, console, task_manager)
    assert success, "Failed to trigger the L.O.V.E. evolution task."

    # Find the task that was just created
    assert task_manager.tasks, "Task manager has no tasks after triggering evolution."
    task_id = max(task_manager.tasks.keys(), key=lambda t: task_manager.tasks[t]['created_at'])
    console.print(f"[bold yellow]Test initiated. Monitoring task {task_id} to increment counter.[/bold yellow]")

    # --- 3. Monitor for Completion ---
    start_time = time.time()
    final_status = None
    pr_url = None

    try:
        while time.time() - start_time < TEST_TIMEOUT:
            task = task_manager.tasks.get(task_id)
            if not task:
                # Task might have been replaced due to a merge conflict
                all_tasks = task_manager.get_status()
                new_task = next((t for t in all_tasks if t['request'] == request and t['status'] != 'merge_failed'), None)
                if new_task:
                    console.print(f"[bold yellow]Task {task_id} was superseded. Now monitoring new task {new_task['id']}.[/bold yellow]")
                    task_id = new_task['id']
                    task = new_task
                else:
                     pytest.fail(f"Original task {task_id} disappeared and no replacement was found.")

            if task.get('pr_url'):
                pr_url = task['pr_url']

            if task['status'] == 'completed':
                final_status = task['status']
                console.print(f"[bold green]Success! Task {task_id} completed.[/bold green]")
                break
            elif task['status'] in ['failed', 'merge_failed']:
                if task['status'] == 'failed':
                    pytest.fail(f"Task {task_id} failed with message: {task['message']}")

            console.print(f"Waiting for task completion... Current status for task {task_id}: {task['status']}")
            time.sleep(POLL_INTERVAL)

        assert final_status == 'completed', f"Test timed out waiting for task completion. Final status: {final_status}"

        # --- 4. Verify the Change on Main Branch ---
        console.print("[bold cyan]Verifying final state of the counter file...[/bold cyan]")
        subprocess.check_call(["git", "checkout", "main"])
        subprocess.check_call(["git", "pull", "origin", "main"])

        assert os.path.exists(counter_file), "Counter file disappeared after merge."
        with open(counter_file, "r") as f:
            final_value = int(f.read().strip())

        console.print(f"[bold cyan]Final counter value is: {final_value}[/bold cyan]")
        assert final_value == initial_value + 1, f"Counter was not incremented correctly. Expected {initial_value + 1}, got {final_value}."
        console.print("[bold green]Verification successful! The counter was incremented correctly.[/bold green]")

    finally:
        # --- 5. Cleanup ---
        if pr_url:
            pr_number_match = re.search(r'/pull/(\d+)', pr_url)
            if pr_number_match:
                pr_number = pr_number_match.group(1)
                branch_name = _get_pr_branch_name(pr_url)
                if branch_name:
                    console.print(f"[bold yellow]Cleaning up branch: {branch_name}[/bold yellow]")
                    github_token = os.environ.get("GITHUB_TOKEN")
                    headers = {"Authorization": f"token {github_token}", "Accept": "application/vnd.github.v3+json"}
                    _delete_pr_branch(repo_owner, repo_name, pr_number, headers)
                    subprocess.run(["git", "branch", "-D", branch_name], capture_output=True)
                    console.print(f"[bold green]Cleanup complete for branch {branch_name}.[/bold green]")

@pytest.mark.real_api
@pytest.mark.asyncio
async def test_jules_merge_conflict_retry_logic(task_manager, console, mocker):
    """
    Tests that the LoveTaskManager correctly retries a task up to 3 times
    when faced with a persistent merge conflict, and then fails.
    """
    request = "Create a new file named 'test_retry.txt' with the content 'hello'."

    # --- Mock GitHub API to always return a merge conflict ---
    mock_response = mocker.Mock()
    mock_response.status_code = 405  # 405 Method Not Allowed indicates a merge conflict
    mocker.patch('love.requests.put', return_value=mock_response)

    # --- Trigger the task ---
    success = trigger_jules_evolution(request, console, task_manager)
    assert success, "Failed to trigger the L.O.V.E. evolution task."

    # --- Monitor the retry logic ---
    start_time = time.time()
    final_status = None
    task_id = max(task_manager.tasks.keys(), key=lambda t: task_manager.tasks[t]['created_at'])

    retry_count = 0
    max_retries = 3

    try:
        while time.time() - start_time < TEST_TIMEOUT:
            task = task_manager.tasks.get(task_id)

            if not task:
                all_tasks = task_manager.get_status()
                # Find the next task in the retry chain
                new_task = next((t for t in all_tasks if t['request'] == request and t.get('retries', 0) == retry_count + 1), None)
                if new_task:
                    retry_count += 1
                    console.print(f"[bold yellow]Merge conflict triggered retry. Now monitoring new task {new_task['id']} (Attempt {retry_count}).[/bold yellow]")
                    task_id = new_task['id']
                    task = new_task
                else:
                    # If we can't find the next task, something is wrong, or the process is complete
                    pass

            if task and task['status'] == 'merge_failed':
                final_status = task['status']
                console.print(f"[bold green]Success! Task {task_id} failed with merge_failed status after {retry_count} retries.[/bold green]")
                break

            # This is a short test, so we can poll more frequently
            time.sleep(10)

        assert final_status == 'merge_failed', f"Test timed out. Final status: {final_status}"
        assert retry_count == max_retries, f"Expected {max_retries} retries, but {retry_count} occurred."

    finally:
        # No cleanup needed as no PR was ever actually merged.
        pass