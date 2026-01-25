
import pytest
import os
from core.integration.dashboard import EvolutionDashboard

@pytest.fixture
def dashboard(tmp_path):
    log_file = tmp_path / "EVOLUTION_LOG.md"
    return EvolutionDashboard(log_file=str(log_file))

def test_log_creation(dashboard):
    assert os.path.exists(dashboard.log_file)
    with open(dashboard.log_file, "r") as f:
        content = f.read()
        assert "# Evolution Dashboard Log" in content
        assert "| Timestamp | Function | Status | Details |" in content

def test_log_evolution_append(dashboard):
    dashboard.log_evolution("my_func", "SUCCESS", "Optimized loop")
    
    with open(dashboard.log_file, "r") as f:
        lines = f.readlines()
        
    # Check last line
    last_line = lines[-1]
    assert "| my_func | SUCCESS | Optimized loop |" in last_line
    # Timestamp should be there too
    assert last_line.count("|") == 5 # Start, Time, Func, Status, Details, End

def test_log_evolution_sanitize(dashboard):
    dashboard.log_evolution("func", "FAIL", "Error\nNew line | Pipe")
    
    with open(dashboard.log_file, "r") as f:
        last_line = f.readlines()[-1]
        
    assert "Error New line \\| Pipe" in last_line
    assert "\n" not in last_line[:-1] # Should be single line
