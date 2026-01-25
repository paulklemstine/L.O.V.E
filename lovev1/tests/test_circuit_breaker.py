
import unittest
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unittest.mock import patch
from core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException

class TestCircuitBreaker(unittest.TestCase):
    @patch('random.uniform', return_value=0)
    def test_circuit_breaker_logic(self, mock_random):
        # Create a breaker with 3 failures threshold and small timeout
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        # Define a function that always fails
        def always_fails():
            raise ValueError("Failed!")
            
        # 1. Fail 3 times -> State should be OPEN
        for i in range(3):
            with self.assertRaises(ValueError):
                breaker.call(always_fails)
                
        self.assertEqual(breaker.failure_count, 3)
        
        # 4. Call again -> Should raise CircuitBreakerOpenException immediately
        # (Elapsed ~0 < 1)
        with self.assertRaises(CircuitBreakerOpenException):
             breaker.call(always_fails)
             
        # 5. Wait for timeout
        time.sleep(1.1)
        
        # 6. Call again -> Should be HALF-OPEN (fails again)
        # Timeout passed. State becomes HALF_OPEN. Function called. Fails.
        # State becomes OPEN. 
        with self.assertRaises(ValueError):
             breaker.call(always_fails)
             
        # Should be back to OPEN
        self.assertEqual(breaker.state.value, "OPEN")
        # And consecutive trips increased?
        self.assertEqual(breaker.consecutive_trips, 1) # started 0, +1
        
        # 7. Test Success
        # Wait again. 
        # Backoff: 1 * 2^1 = 2s. Jitter=0. Total=2s.
        time.sleep(2.1) 
        
        def success():
            return "Success"
            
        # Run success
        res = breaker.call(success)
        self.assertEqual(res, "Success")
        self.assertEqual(breaker.state.value, "CLOSED")
        self.assertEqual(breaker.failure_count, 0)
        
if __name__ == '__main__':
    unittest.main()
