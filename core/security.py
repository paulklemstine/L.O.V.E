"""
Security module for L.O.V.E. system.
Provides creator instance verification for secure operations like code injection.
"""
import os
from pathlib import Path

# Default private key filename to check for
CREATOR_KEY_FILENAME = "creator_private_key.pem"

# Cache the result to avoid repeated filesystem checks
_is_creator_cache = None


def is_creator_instance(key_path: str = None) -> bool:
    """
    Check if this is a creator instance by verifying the presence of
    the creator's private key file.
    
    Args:
        key_path: Optional custom path to the private key file.
                  Defaults to checking for 'creator_private_key.pem' in project root.
    
    Returns:
        True if the private key file exists, False otherwise.
    """
    global _is_creator_cache
    
    # Use cached result if available and no custom path specified
    if _is_creator_cache is not None and key_path is None:
        return _is_creator_cache
    
    # Determine the path to check
    if key_path:
        check_path = Path(key_path)
    else:
        # Check in project root (where love.py is located)
        project_root = Path(__file__).parent.parent
        check_path = project_root / CREATOR_KEY_FILENAME
    
    # Check if the file exists
    result = check_path.exists() and check_path.is_file()
    
    # Cache the result for default path
    if key_path is None:
        _is_creator_cache = result
    
    return result


def clear_creator_cache():
    """
    Clear the cached creator instance check result.
    Useful for testing or when the key file status may have changed.
    """
    global _is_creator_cache
    _is_creator_cache = None


def get_creator_key_path() -> Path:
    """
    Get the expected path to the creator's private key file.
    
    Returns:
        Path object pointing to the expected key file location.
    """
    project_root = Path(__file__).parent.parent
    return project_root / CREATOR_KEY_FILENAME


def verify_injection_allowed() -> tuple[bool, str]:
    """
    Check if code injection is allowed on this instance.
    
    Returns:
        Tuple of (allowed: bool, message: str)
    """
    if is_creator_instance():
        return True, "Creator instance verified. Code injection allowed."
    else:
        key_path = get_creator_key_path()
        return False, f"Code injection denied. Creator key not found at: {key_path}"
