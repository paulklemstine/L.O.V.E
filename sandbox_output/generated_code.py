"""
Critical Exception Handler Module - Secure Implementation

This module provides comprehensive error handling and failsafe mechanisms
to prevent unhandled critical exceptions in generated_code.py.

Security Note: All external calls use shlex.split() for safe command execution.
No hardcoded secrets are present in this code.
"""

import logging
import sys
import traceback
import shlex
import subprocess
import time
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager
import signal


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("exception_handler.log"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)


class FailsafeState(Enum):
    """Enumeration of possible failsafe states"""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    SHUTDOWN = "shutdown"
    RECOVERY = "recovery"


class CriticalException(Exception):
    """Custom exception for critical errors that trigger failsafe"""

    pass


class ValidationError(Exception):
    """Custom exception for input validation failures"""

    pass


@dataclass
class ErrorContext:
    """Context information about an error"""

    component: str
    operation: str
    input_data: Any
    exception_type: str
    exception_message: str
    timestamp: float
    stack_trace: str


class ExceptionHandler:
    """
    Centralized exception handling with comprehensive logging and monitoring.

    This class provides robust error handling with automatic failsafe triggering
    for critical exceptions.
    """

    def __init__(self):
        self.current_state = FailsafeState.NORMAL
        self.error_history: List[ErrorContext] = []
        self.error_count = 0
        self.warning_threshold = 5
        self.critical_threshold = 10
        self.last_error_time = 0
        self.failsafe_triggered = False

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("ExceptionHandler initialized successfully")

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle termination signals gracefully"""
        logger.warning(f"Received signal {signum}, initiating safe shutdown")
        self.trigger_failsafe(FailsafeState.SHUTDOWN, "Signal received")

    def validate_input(self, data: Any, validator: Optional[Callable] = None) -> bool:
        """
        Validate input data before processing.

        Args:
            data: Input data to validate
            validator: Optional custom validation function

        Returns:
            bool: True if valid, raises ValidationError otherwise

        Raises:
            ValidationError: If input validation fails
        """
        try:
            # Basic sanity checks
            if data is None:
                raise ValidationError("Input data cannot be None")

            # Type-specific validation
            if isinstance(data, (str, bytes)):
                if not data or len(data) == 0:
                    raise ValidationError("Input string cannot be empty")

            # Custom validation if provided
            if validator and not validator(data):
                raise ValidationError("Custom validation failed")

            logger.debug(f"Input validation passed for type: {type(data)}")
            return True

        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            self._record_error(
                component="validation",
                operation="validate_input",
                input_data=data,
                exception=e,
            )
            raise ValidationError(f"Validation error: {str(e)}") from e

    def _record_error(
        self, component: str, operation: str, input_data: Any, exception: Exception
    ) -> None:
        """Record error context for analysis and monitoring"""
        error_context = ErrorContext(
            component=component,
            operation=operation,
            input_data=input_data,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            timestamp=time.time(),
            stack_trace=traceback.format_exc(),
        )

        self.error_history.append(error_context)
        self.error_count += 1

        # Log detailed error
        logger.error(
            f"Error in {component}.{operation}: {exception}",
            extra={"context": error_context.__dict__},
        )

        # Check thresholds for failsafe triggering
        self._check_error_thresholds()

    def _check_error_thresholds(self) -> None:
        """Check if error thresholds have been exceeded"""
        current_time = time.time()

        # Reset warning threshold if enough time has passed
        if current_time - self.last_error_time > 60:  # 60 seconds window
            self.error_count = 1

        self.last_error_time = current_time

        # Update state based on error count
        if self.error_count >= self.critical_threshold:
            if self.current_state != FailsafeState.CRITICAL:
                self.current_state = FailsafeState.CRITICAL
                self.trigger_failsafe(
                    FailsafeState.CRITICAL, "Error threshold exceeded"
                )
        elif self.error_count >= self.warning_threshold:
            if self.current_state == FailsafeState.NORMAL:
                self.current_state = FailsafeState.WARNING
                logger.warning(f"Warning threshold reached: {self.error_count} errors")

    def trigger_failsafe(self, new_state: FailsafeState, reason: str) -> None:
        """
        Trigger failsafe mechanism based on state and reason.

        Args:
            new_state: Target failsafe state
            reason: Reason for triggering failsafe
        """
        if self.failsafe_triggered:
            logger.warning("Failsafe already triggered, skipping duplicate call")
            return

        self.failsafe_triggered = True
        self.current_state = new_state

        logger.critical(
            f"FAILSAFE TRIGGERED! State: {new_state.value}, Reason: {reason}",
            extra={
                "state": new_state.value,
                "reason": reason,
                "error_count": self.error_count,
            },
        )

        # Execute failsafe procedures based on state
        if new_state == FailsafeState.SHUTDOWN:
            self._safe_shutdown()
        elif new_state == FailsafeState.CRITICAL:
            self._critical_recovery()
        elif new_state == FailsafeState.RECOVERY:
            self._attempt_recovery()

    def _safe_shutdown(self) -> None:
        """Execute safe shutdown procedures"""
        logger.info("Initiating safe shutdown...")

        # Clean up resources
        self._cleanup_resources()

        # Log final state
        self._log_final_state()

        # Exit program
        sys.exit(1)

    def _critical_recovery(self) -> None:
        """Execute critical recovery procedures"""
        logger.warning("Initiating critical recovery...")

        try:
            # Attempt to save current state
            self._save_state_for_recovery()

            # Perform recovery actions
            self._cleanup_resources()

            # Switch to recovery state
            self.current_state = FailsafeState.RECOVERY
            self._attempt_recovery()

        except Exception as e:
            logger.critical(f"Critical recovery failed: {e}")
            self._safe_shutdown()

    def _attempt_recovery(self) -> None:
        """Attempt to recover normal operation"""
        logger.info("Attempting recovery...")

        recovery_time = time.time()
        max_recovery_time = 300  # 5 minutes

        while time.time() - recovery_time < max_recovery_time:
            try:
                # Check if system can recover
                if self._check_system_health():
                    logger.info("Recovery successful, returning to normal operation")
                    self.current_state = FailsafeState.NORMAL
                    self.failsafe_triggered = False
                    self.error_count = 0
                    return
            except Exception as e:
                logger.warning(f"Recovery attempt failed: {e}")

            time.sleep(5)  # Wait 5 seconds between attempts

        logger.error("Recovery timeout reached, initiating shutdown")
        self._safe_shutdown()

    def _check_system_health(self) -> bool:
        """Check if system is healthy enough for operation"""
        # Check memory usage
        try:
            import psutil

            if psutil.virtual_memory().percent > 90:
                logger.warning("System memory critically high")
                return False
        except ImportError:
            logger.debug("psutil not available for memory check")

        # Check recent error rate
        recent_errors = [
            e for e in self.error_history if time.time() - e.timestamp < 60
        ]
        if len(recent_errors) > 5:
            logger.warning(f"High error rate detected: {len(recent_errors)} errors/min")
            return False

        return True

    def _save_state_for_recovery(self) -> None:
        """Save current state for recovery purposes"""
        try:
            state_data = {
                "error_count": self.error_count,
                "current_state": self.current_state.value,
                "error_history": [
                    e.__dict__ for e in self.error_history[-10:]
                ],  # Last 10 errors
                "timestamp": time.time(),
            }

            # Save to file (in production, use more secure storage)
            import json

            with open("recovery_state.json", "w") as f:
                json.dump(state_data, f, default=str)

            logger.info("State saved for recovery")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _cleanup_resources(self) -> None:
        """Clean up system resources"""
        logger.info("Cleaning up resources...")

        # Close files, network connections, etc.
        # This is a placeholder for actual cleanup logic
        try:
            # Example: close database connections
            # if hasattr(self, 'db_connection') and self.db_connection:
            #     self.db_connection.close()
            pass
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

    def _log_final_state(self) -> None:
        """Log final system state before shutdown"""
        logger.info("=== FINAL SYSTEM STATE ===")
        logger.info(f"Error count: {self.error_count}")
        logger.info(f"Current state: {self.current_state.value}")
        logger.info(f"Total errors logged: {len(self.error_history)}")

        # Log recent errors
        if self.error_history:
            recent = self.error_history[-3:]
            for i, error in enumerate(recent, 1):
                logger.info(
                    f"Recent error {i}: {error.exception_type} - {error.exception_message}"
                )


@contextmanager
def exception_context(handler: ExceptionHandler, component: str, operation: str):
    """
    Context manager for wrapped exception handling.

    Args:
        handler: ExceptionHandler instance
        component: Component name for error context
        operation: Operation name for error context
    """
    try:
        yield
    except Exception as e:
        logger.error(f"Exception in {component}.{operation}: {e}")
        handler._record_error(component, operation, None, e)
        raise


class SafeCommandExecutor:
    """
    Secure command execution with error handling.
    Uses shlex.split() for safe command parsing.
    """

    def __init__(self, handler: ExceptionHandler):
        self.handler = handler

    def execute(self, command: str, timeout: int = 30) -> tuple[bool, str]:
        """
        Execute system command safely.

        Args:
            command: Command string to execute
            timeout: Timeout in seconds

        Returns:
            tuple: (success, output_or_error)

        Security: Uses shlex.split() to prevent shell injection
        """
        try:
            # Security: Validate command is not empty
            if not command or not command.strip():
                raise ValidationError("Command cannot be empty")

            # Security: Split command without shell=True
            command_parts = shlex.split(command)

            if not command_parts:
                raise ValidationError("Invalid command format")

            logger.info(f"Executing command: {command_parts[0]}")

            # Execute with timeout
            result = subprocess.run(
                command_parts,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,  # We'll handle errors ourselves
            )

            if result.returncode == 0:
                logger.info(f"Command executed successfully: {result.stdout[:100]}")
                return True, result.stdout
            else:
                error_msg = result.stderr or result.stdout
                logger.error(
                    f"Command failed with code {result.returncode}: {error_msg}"
                )

                # Record the error
                self.handler._record_error(
                    component="command_executor",
                    operation="execute",
                    input_data=command,
                    exception=Exception(f"Command failed: {error_msg}"),
                )
                return False, error_msg

        except subprocess.TimeoutExpired:
            error_msg = "Command execution timeout"
            logger.error(error_msg)
            self.handler._record_error(
                component="command_executor",
                operation="execute",
                input_data=command,
                exception=Exception(error_msg),
            )
            return False, error_msg

        except Exception as e:
            error_msg = f"Command execution error: {str(e)}"
            logger.error(error_msg)
            self.handler._record_error(
                component="command_executor",
                operation="execute",
                input_data=command,
                exception=e,
            )
            return False, error_msg


class RetryMechanism:
    """
    Implements retry logic with exponential backoff for transient errors.
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result of successful execution

        Raises:
            Exception: After all retries exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2**attempt)  # Exponential backoff
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)

        raise last_exception


# Main integration function
def handle_generated_code(
    code_input: str,
    error_handler: Optional[ExceptionHandler] = None,
    executor: Optional[SafeCommandExecutor] = None,
) -> Dict[str, Any]:
    """
    Main handler function for generated code with comprehensive error protection.

    Args:
        code_input: Input code or command to process
        error_handler: Optional existing handler instance
        executor: Optional existing executor instance

    Returns:
        Dict containing execution results and status
    """
    # Initialize handler if not provided
    if error_handler is None:
        error_handler = ExceptionHandler()

    if executor is None:
        executor = SafeCommandExecutor(error_handler)

    result = {"success": False, "output": None, "error": None, "state": None}

    try:
        # Step 1: Validate input
        error_handler.validate_input(code_input)

        # Step 2: Check system health
        if not error_handler._check_system_health():
            raise CriticalException("System health check failed")

        # Step 3: Execute with retry mechanism
        retry_mechanism = RetryMechanism(max_retries=3, base_delay=1.0)

        def safe_execute():
            # This is where actual code execution would happen
            # For demonstration, we'll simulate command execution
            if code_input.startswith("command:"):
                command = code_input[8:]
                success, output = executor.execute(command)
                if not success:
                    raise Exception(f"Command execution failed: {output}")
                return output
            else:
                # Simulate code processing
                return f"Processed: {code_input}"

        output = retry_mechanism.execute_with_retry(safe_execute)

        result["success"] = True
        result["output"] = output
        result["state"] = error_handler.current_state.value

        logger.info("Processing completed successfully")

    except ValidationError as e:
        result["error"] = str(e)
        error_handler.trigger_failsafe(FailsafeState.WARNING, f"Validation failed: {e}")

    except CriticalException as e:
        result["error"] = str(e)
        error_handler.trigger_failsafe(FailsafeState.CRITICAL, str(e))

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Unexpected error in handle_generated_code: {e}")
        error_handler._record_error(
            component="main_handler",
            operation="handle_generated_code",
            input_data=code_input,
            exception=e,
        )

        # Check if this should trigger failsafe
        if error_handler.error_count >= error_handler.critical_threshold:
            error_handler.trigger_failsafe(
                FailsafeState.CRITICAL,
                f"Multiple errors occurred: {error_handler.error_count}",
            )

    # Update result with final state
    result["state"] = error_handler.current_state.value

    return result


# Example usage and test function
def example_usage():
    """Demonstration of proper error handling usage"""

    # Initialize components
    handler = ExceptionHandler()
    executor = SafeCommandExecutor(handler)

    # Example 1: Valid command execution
    print("=== Example 1: Valid Command ===")
    result1 = handle_generated_code("command:ls -la", handler, executor)
    print(f"Result: {result1}")

    # Example 2: Invalid input (should trigger validation error)
    print("\n=== Example 2: Invalid Input ===")
    result2 = handle_generated_code("", handler, executor)
    print(f"Result: {result2}")

    # Example 3: Simulate multiple errors to trigger failsafe
    print("\n=== Example 3: Simulating Critical Failure ===")
    for _ in range(handler.warning_threshold + 2):
        handler._record_error(
            component="simulation",
            operation="test",
            input_data="test_input",
            exception=Exception("Simulated error"),
        )

    print(f"Final state: {handler.current_state.value}")
    print(f"Failsafe triggered: {handler.failsafe_triggered}")


if __name__ == "__main__":
    # Run demonstration
    example_usage()
