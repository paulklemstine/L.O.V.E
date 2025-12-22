
import pytest
from unittest import mock
from core.integration.git_handler import GitHandler

@pytest.fixture
def git():
    return GitHandler(repo_path="/tmp/repo")

def test_branch_exists(git):
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "  main\n* agent/evolution-experiments"
        assert git.branch_exists("agent/evolution-experiments")
        
        mock_run.return_value.stdout = "  main"
        assert not git.branch_exists("missing-branch")

def test_create_branch_new(git):
    with mock.patch("subprocess.run") as mock_run:
        # First call checks existence (returns false), second creates
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout="  main"), # list
            mock.Mock(returncode=0, stdout="", stderr="") # create
        ]
        
        assert git.create_branch("new-branch")
        
        # Check calls
        assert mock_run.call_count == 2
        args1, _ = mock_run.call_args_list[0]
        assert args1[0] == ["git", "branch", "--list", "new-branch"]
        args2, _ = mock_run.call_args_list[1]
        assert args2[0] == ["git", "branch", "new-branch"]

def test_checkout_branch(git):
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        assert git.checkout_branch("feature")
        args, _ = mock_run.call_args
        assert args[0] == ["git", "checkout", "feature"]

def test_commit_changes(git):
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        assert git.commit_changes("Update")
        
        # Expect add and commit
        assert mock_run.call_count == 2
        args1, _ = mock_run.call_args_list[0]
        assert args1[0] == ["git", "add", "."]
        args2, _ = mock_run.call_args_list[1]
        assert args2[0] == ["git", "commit", "-m", "Update"]

def test_create_pr_simulated(git):
    # Just checking logic returns True
    assert git.create_pr("Title", "Body")
