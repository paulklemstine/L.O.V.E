import subprocess
import sys

def test_smoke():
    """
    A simple smoke test to ensure the script can be invoked without crashing.
    """
    try:
        # We run with --help, which should trigger the dependency checks and then exit cleanly.
        result = subprocess.run([sys.executable, 'evolve.py', '--help'], capture_output=True, text=True, check=True)
        assert "L.O.V.E." in result.stdout
        assert result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Smoke test failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise