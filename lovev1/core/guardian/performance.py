
import time
import logging
from typing import Tuple

class PerformanceMonitor:
    def __init__(self, sandbox):
        self.sandbox = sandbox

    def measure_execution_time(self, command: str) -> float:
        """
        Measures the wall-clock execution time of a command in the sandbox.
        Returns time in seconds.
        """
        start_time = time.perf_counter()
        
        # We assume command success for measurement reliability? 
        # Or do we return time regardless?
        # If command fails, the time might be short (fast fail).
        # We should probably check exit code.
        exit_code, stdout, stderr = self.sandbox.run_command(command)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        if exit_code != 0:
            logging.warning(f"Command '{command}' failed during performance measurement. Duration: {duration:.4f}s")
            # We still return the duration, but user should be aware it failed.
            # The upstream caller (verification pipeline) handles semantics. 
            # This is just for perf.
            
        logging.info(f"Performance Measurement: '{command}' took {duration:.4f}s")
        return duration

    def check_regression(self, original_time: float, new_time: float, threshold_ratio: float = 1.10) -> bool:
        """
        Checks if new_time is significantly worse than original_time.
        Returns False if regression detected (reject).
        Returns True if acceptable.
        
        Args:
            threshold_ratio: Rejects if new_time > original_time * threshold. Default 1.10 (10% slower).
        """
        if original_time <= 0:
            # If original time is effectively 0 or invalid, we can't really regress strictly?
            # Or infinite regression.
            # Assume acceptable if we have no baseline? Or strict? 
            # If we are optimizing, 0.0s baseline is hard.
            logging.warning("Original time is <= 0. Assuming pass.")
            return True
            
        limit = original_time * threshold_ratio
        if new_time > limit:
            logging.warning(f"Performance Regression Detailed: Old={original_time:.4f}s, New={new_time:.4f}s, Limit={limit:.4f}s")
            return False
            
        return True
