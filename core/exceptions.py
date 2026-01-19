class TemporaryEnvironmentError(Exception):
    """
    Raised when an external environment (like social media or web) is legally reachable
    but temporarily failing (timeouts, 0 results, 503s).
    """
    pass
