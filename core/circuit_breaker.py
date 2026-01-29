import time
import random
import logging
from enum import Enum
from functools import wraps

# Setup logger
logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreakerOpenException(Exception):
    """Raised when the circuit is open and we should fail fast."""
    pass

class CircuitBreaker:
    """
    Implements the Circuit Breaker pattern to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation. Errors count towards threshold.
    - OPEN: Fails fast for a duration.
    - HALF_OPEN: Allows one trial request. Success -> CLOSED, Failure -> OPEN.
    """
    def __init__(self, failure_threshold=3, recovery_timeout=30, exceptions=(Exception,)):
        """
        Args:
            failure_threshold (int): Number of failures before opening the circuit.
            recovery_timeout (int): Base seconds to wait before attempting recovery (HALF-OPEN).
            exceptions (tuple): Tuple of exception types that count as failures.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.exceptions = exceptions
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.consecutive_trips = 0 # How many times we've gone to OPEN consecutively without full recovery

    def _reset(self):
        """Resets the circuit breaker to initial CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.consecutive_trips = 0
        logger.info("CircuitBreaker reset to CLOSED.")

    def call(self, func, *args, **kwargs):
        """
        Executes the function with circuit breaker protection.
        """
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            elapsed = time.time() - self.last_failure_time
            
            # Dynamic duration: base timeout * (2^consecutive_trips) + jitter
            # Cap the backoff to avoid waiting forever (e.g., max 10 minutes)
            backoff_factor = min(self.consecutive_trips, 6) # 2^6 = 64x
            wait_time = self.recovery_timeout * (2 ** backoff_factor)
            
            # Add jitter: random(0, 1000ms) -> 0 to 1s
            jitter = random.uniform(0, 1)
            total_wait = wait_time + jitter
            
            if elapsed < total_wait:
                remaining = total_wait - elapsed
                msg = f"Circuit is OPEN. Fail fast. Retry available in {remaining:.2f}s"
                # logger.debug(msg) # Debug level to avoid spamming logs
                raise CircuitBreakerOpenException(msg)
            
            self.state = CircuitState.HALF_OPEN
            logger.info("CircuitBreaker transitioning to HALF-OPEN. Attempting trial request.")

        # In CLOSED or HALF_OPEN (after timeout), we try the call
        try:
            result = func(*args, **kwargs)
            
            if self.state == CircuitState.HALF_OPEN:
                self._reset() # Success in trial!
            elif self.failure_count > 0:
                # If we were CLOSED but had some failures, a success resets the count?
                # Usually yes, or we decay it. Resetting is simpler.
                self.failure_count = 0
            
            return result
            
        except self.exceptions as e:
            self._handle_failure(e)
            raise e

    def _handle_failure(self, exception):
        """Handles a failure during execution."""
        # Check for 429 (Too Many Requests) specifically
        is_429 = False
        if hasattr(exception, 'status_code') and exception.status_code == 429:
            is_429 = True
        elif hasattr(exception, 'response') and hasattr(exception.response, 'status_code') and exception.response.status_code == 429:
            is_429 = True
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        logger.warning(f"CircuitBreaker caught failure {self.failure_count}/{self.failure_threshold}. Is 429: {is_429}. Exception: {exception}")

        # Trip criteria:
        # 1. It's a 429 (immediate trip)
        # 2. We are in HALF_OPEN (failed trial)
        # 3. Failure count reached threshold
        if is_429 or self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.consecutive_trips += 1
            logger.error(f"CircuitBreaker transitioning to OPEN. Consecutive trips: {self.consecutive_trips}")
