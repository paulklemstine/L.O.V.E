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

from love import LoveTaskManager, trigger_love_evolution, get_git_repo_info
from rich.console import Console
from core.retry import retry

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

# --- Configuration ---
# Ensure JULES_API_KEY and GITHUB_TOKEN are set as environment variables
# You can use a .env file and `python-dotenv` or set them manually:
# export JULES_API_KEY="your_key_here"
# export GITHUB_TOKEN="your_token_here"

# --- Test Setup ---
TEST_TIMEOUT = 900  # 15 minutes, as real-world API calls can be slow
POLL_INTERVAL = 30  # 30 seconds

@pytest.fixture(scope="module")
def console():
    """Provides a Rich Console instance for the test module."""
    return Console()

@pytest.fixture(scope="module")
def task_manager(console):
    """
    Provides a live LoveTaskManager instance. The background loop is allowed
    to run as it would in production.
    """
    manager = LoveTaskManager(console=console)
    manager.start()
    yield manager
    manager.stop()

# Skip this test if the required API keys are not set
JULES_API_KEY = os.environ.get("JULES_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
pytestmark = pytest.mark.skipif(not JULES_API_KEY or not GITHUB_TOKEN, reason="JULES_API_KEY and GITHUB_TOKEN environment variables not set")


@pytest.mark.real_api
@pytest.mark.asyncio
async def test_love_happy_path_lifecycle(task_manager, console):
    """
    Tests the full "happy path" lifecycle of a L.O.V.E. task using real APIs.
    1. Triggers an evolution to add a new test file.
    2. Monitors the task until a pull request is created.
    3. Validates that the pull request is tested and merged automatically.
    4. Cleans up the PR branch.
    """
    # --- 1. Trigger the Evolution Task ---
    # A unique request to avoid collision with other test runs
    request = f"Create a new test file named 'tests/test_autogen_{uuid.uuid4().hex[:8]}.py' with a simple passing test."

    # Use the actual trigger function from love.py
    success = trigger_love_evolution(request, console, task_manager)
    assert success, "Failed to trigger the L.O.V.E. evolution task via the L.O.V.E. API."

    # Find the task that was just created
    assert task_manager.tasks, "Task manager has no tasks after triggering evolution."
    task_id = max(task_manager.tasks.keys(), key=lambda t: task_manager.tasks[t]['created_at'])
    console.print(f"[bold yellow]Test initiated. Monitoring task {task_id} for request: '{request}'[/bold yellow]")

    # --- 2. Monitor for PR Creation ---
    start_time = time.time()
    pr_url = None
    while time.time() - start_time < TEST_TIMEOUT:
        task = task_manager.tasks.get(task_id)
        if task and task.get('pr_url'):
            pr_url = task['pr_url']
            console.print(f"[bold green]Success! PR Created: {pr_url}[/bold green]")
            break
        elif task and task['status'] in ['failed', 'merge_failed']:
            pytest.fail(f"Task {task_id} failed with message: {task['message']}")

        console.print(f"Waiting for PR... Current status: {task['status'] if task else 'N/A'}")
        time.sleep(POLL_INTERVAL)

    assert pr_url, f"Test timed out after {TEST_TIMEOUT}s waiting for a pull request."

    # --- 3. Wait for Merge Completion ---
    start_time = time.time()
    final_status = None
    branch_name = None

    # We need the branch name for cleanup, so we get it from the PR URL
    repo_owner, repo_name = get_git_repo_info()
    pr_number_match = re.search(r'/pull/(\d+)', pr_url)
    assert pr_number_match, f"Could not extract PR number from URL: {pr_url}"
    pr_number = pr_number_match.group(1)
    branch_name = _get_pr_branch_name(pr_url)
    assert branch_name, f"Could not determine branch name from PR URL {pr_url}"

    try:
        while time.time() - start_time < TEST_TIMEOUT:
            task = task_manager.tasks.get(task_id)
            if task and task['status'] == 'completed':
                final_status = task['status']
                console.print(f"[bold green]Success! Task {task_id} completed.[/bold green]")
                break
            elif task and task['status'] in ['failed', 'merge_failed']:
                pytest.fail(f"Task {task_id} failed during merge process with message: {task['message']}")

            console.print(f"Waiting for merge... Current status: {task['status'] if task else 'N/A'}")
            time.sleep(POLL_INTERVAL)

        assert final_status == 'completed', f"Test timed out waiting for PR merge. Final status: {final_status}"

        # --- 4. Verify the new file exists on the main branch ---
        # Pull the latest changes and check for the file
        subprocess.check_call(["git", "checkout", "main"])
        subprocess.check_call(["git", "pull"])

        # The exact filename is unknown, but we can list the directory and find the new file
        files_in_tests = os.listdir("tests")
        autogen_files = [f for f in files_in_tests if f.startswith("test_autogen_")]
        assert autogen_files, "The new autogenerated test file was not found on the main branch after merge."

        # Clean up the created file
        for f in autogen_files:
            os.remove(os.path.join("tests", f))

    finally:
        # --- 5. Cleanup ---
        # This block ensures the PR branch is deleted even if the test fails.
        if branch_name:
            console.print(f"[bold yellow]Cleaning up branch: {branch_name}[/bold yellow]")
            github_token = os.environ.get("GITHUB_TOKEN")
            headers = {"Authorization": f"token {github_token}", "Accept": "application/vnd.github.v3+json"}
            # Using the actual delete function for cleanup
            _delete_pr_branch(repo_owner, repo_name, pr_number, headers)

            # Also attempt to delete the local branch if it exists
            subprocess.run(["git", "branch", "-D", branch_name])

            console.print(f"[bold green]Cleanup complete.[/bold green]")

@pytest.mark.real_api
@pytest.mark.asyncio
async def test_love_merge_conflict_resolution_lifecycle(task_manager, console):
    """
    Tests the full lifecycle of a L.O.V.E. task with a merge conflict.
    1. Triggers an evolution to modify an existing file.
    2. After the PR is created, manually introduce a conflicting change to the main branch.
    3. Validates that the merge conflict is detected and resolved.
    4. Cleans up the PR branch.
    """
    # --- 1. Setup: Create a file to be modified ---
    conflict_file = f"tests/test_conflict_{uuid.uuid4().hex[:8]}.py"
    with open(conflict_file, "w") as f:
        f.write("# Original line\n")

    subprocess.check_call(["git", "add", conflict_file])
    subprocess.check_call(["git", "commit", "-m", f"feat: add conflict file {conflict_file}"])
    subprocess.check_call(["git", "push"])

    # --- 2. Trigger the Evolution Task ---
    request = f"In the file `{conflict_file}`, change the comment to '# Modified by L.O.V.E.'"

    success = trigger_love_evolution(request, console, task_manager)
    assert success, "Failed to trigger the L.O.V.E. evolution task."

    task_id = max(task_manager.tasks.keys(), key=lambda t: task_manager.tasks[t]['created_at'])
    console.print(f"[bold yellow]Conflict Test initiated. Monitoring task {task_id}[/bold yellow]")

    # --- 3. Monitor for PR Creation ---
    start_time = time.time()
    pr_url = None
    while time.time() - start_time < TEST_TIMEOUT:
        task = task_manager.tasks.get(task_id)
        if task and task.get('pr_url'):
            pr_url = task['pr_url']
            break
        time.sleep(POLL_INTERVAL)

    assert pr_url, "Test timed out waiting for a pull request."
    console.print(f"[bold green]PR Created for conflict test: {pr_url}[/bold green]")

    # --- 4. Introduce a merge conflict ---
    with open(conflict_file, "w") as f:
        f.write("# Modified by main branch\n")
    subprocess.check_call(["git", "add", conflict_file])
    subprocess.check_call(["git", "commit", "-m", "chore: create conflicting change"])
    subprocess.check_call(["git", "push"])

    console.print("[bold yellow]Conflicting change pushed to main.[/bold yellow]")

    # --- 5. Wait for Merge Completion ---
    start_time = time.time()
    final_status = None
    repo_owner, repo_name = get_git_repo_info()
    pr_number_match = re.search(r'/pull/(\d+)', pr_url)
    pr_number = pr_number_match.group(1)
    branch_name = _get_pr_branch_name(pr_url)

    try:
        while time.time() - start_time < TEST_TIMEOUT:
            task = task_manager.tasks.get(task_id)
            if task and task['status'] == 'completed':
                final_status = task['status']
                break
            elif task and task['status'] in ['failed', 'merge_failed']:
                 pytest.fail(f"Task {task_id} failed with message: {task['message']}")
            time.sleep(POLL_INTERVAL)

        assert final_status == 'completed', "Test timed out waiting for merge conflict resolution."
        console.print("[bold green]Merge conflict successfully resolved and merged.[/bold green]")

        # --- 6. Verify the resolution ---
        subprocess.check_call(["git", "pull"])
        with open(conflict_file, "r") as f:
            content = f.read()
        # The LLM should have combined the changes. We look for the word "L.O.V.E.".
        assert "L.O.V.E." in content, "Resolved file content does not seem to contain the LLM's resolution."

    finally:
        # --- 7. Cleanup ---
        if branch_name:
            console.print(f"[bold yellow]Cleaning up branch: {branch_name}[/bold yellow]")
            github_token = os.environ.get("GITHUB_TOKEN")
            headers = {"Authorization": f"token {github_token}", "Accept": "application/vnd.github.v3+json"}
            _delete_pr_branch(repo_owner, repo_name, pr_number, headers)
            subprocess.run(["git", "branch", "-D", branch_name])

        if os.path.exists(conflict_file):
            os.remove(conflict_file)

        # Commit the deletion
        subprocess.run(["git", "add", "-u"])
        subprocess.run(["git", "commit", "-m", "chore: clean up conflict test file"])
        subprocess.run(["git", "push"])

        console.print(f"[bold green]Cleanup complete.[/bold green]")