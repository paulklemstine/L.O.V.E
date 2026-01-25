
import pytest
from unittest import mock
import time
from core.guardian.performance import PerformanceMonitor

class MockSandbox:
    def run_command(self, cmd):
        # We can simulate delay here if we didn't mock time.perf_counter
        return 0, "", ""

@pytest.fixture
def monitor():
    return PerformanceMonitor(sandbox=MockSandbox())

def test_measure_execution_time(monitor):
    # Mock time.perf_counter to simulate 1.5 seconds duration
    with mock.patch("time.perf_counter", side_effect=[100.0, 101.5]):
        duration = monitor.measure_execution_time("echo test")
        assert duration == 1.5

def test_check_regression_pass(monitor):
    # Old = 1.0, New = 1.05 (5% slower) -> Pass (limit 1.10)
    assert monitor.check_regression(1.0, 1.05)

def test_check_regression_fail(monitor):
    # Old = 1.0, New = 1.11 (11% slower) -> Fail
    assert not monitor.check_regression(1.0, 1.11)

def test_check_regression_exact_limit(monitor):
    # Old = 1.0, New = 1.10 -> Pass (<= limit)
    # My implementation uses strictly >, so 1.10 is NOT > 1.10
    assert monitor.check_regression(1.0, 1.10)

def test_measure_execution_time_uses_sandbox(monitor):
    with mock.patch.object(monitor.sandbox, 'run_command', return_value=(0, "", "")) as mock_run:
        monitor.measure_execution_time("ls -la")
        mock_run.assert_called_once_with("ls -la")
