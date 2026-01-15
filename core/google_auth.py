"""
Google OAuth Authentication Module for L.O.V.E.

Handles OAuth 2.0 token generation and refresh for Google APIs (Jules, etc.)
Tokens are cached and auto-refreshed on startup.
"""

import os
import subprocess
import json
import time
import logging
from pathlib import Path
from typing import Optional, Tuple

# Token cache file location
TOKEN_CACHE_FILE = Path(__file__).parent.parent / ".google_oauth_cache.json"
TOKEN_EXPIRY_BUFFER = 300  # Refresh 5 minutes before expiry

# Quota project for Jules API (required for user credentials)
# This should be a project where the user has billing enabled
QUOTA_PROJECT = "gen-lang-client-0917722541"  # L.O.V.E. project


class GoogleAuthManager:
    """Manages Google OAuth 2.0 tokens for L.O.V.E."""
    
    def __init__(self):
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[float] = None
        self._load_cached_token()
    
    def _load_cached_token(self):
        """Load token from cache file if valid."""
        if TOKEN_CACHE_FILE.exists():
            try:
                with open(TOKEN_CACHE_FILE, 'r') as f:
                    cache = json.load(f)
                    expiry = cache.get('expiry', 0)
                    if expiry > time.time() + TOKEN_EXPIRY_BUFFER:
                        self._access_token = cache.get('access_token')
                        self._token_expiry = expiry
                        logging.info(f"Loaded valid OAuth token from cache. Expires in {int(expiry - time.time())}s")
                    else:
                        logging.debug(f"Cached token expired. Expiry: {expiry}, Now: {time.time()}")
            except Exception as e:
                logging.warning(f"Failed to load token cache: {e}")
    
    def _save_token_cache(self):
        """Save token to cache file."""
        if self._access_token and self._token_expiry:
            try:
                with open(TOKEN_CACHE_FILE, 'w') as f:
                    json.dump({
                        'access_token': self._access_token,
                        'expiry': self._token_expiry
                    }, f)
                # Secure the file
                os.chmod(TOKEN_CACHE_FILE, 0o600)
            except Exception as e:
                logging.warning(f"Failed to save token cache: {e}")
    
    def _check_gcloud_installed(self) -> bool:
        """Check if gcloud CLI is installed."""
        try:
            result = subprocess.run(
                ['gcloud', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _install_gcloud(self) -> bool:
        """Attempt to install gcloud CLI (Linux only)."""
        logging.info("Attempting to install Google Cloud CLI...")
        try:
            # Add Google Cloud repo and install
            commands = [
                'curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg 2>/dev/null || true',
                'echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list',
                'sudo apt-get update -qq',
                'sudo apt-get install -y -qq google-cloud-cli'
            ]
            
            for cmd in commands:
                result = subprocess.run(cmd, shell=True, capture_output=True, timeout=300)
                if result.returncode != 0 and 'apt-get install' in cmd:
                    logging.error(f"gcloud install failed: {result.stderr.decode()}")
                    return False
            
            logging.info("Google Cloud CLI installed successfully.")
            return True
        except Exception as e:
            logging.error(f"Failed to install gcloud: {e}")
            return False
    
    def _check_gcloud_authenticated(self) -> bool:
        """Check if gcloud has valid credentials."""
        try:
            result = subprocess.run(
                ['gcloud', 'auth', 'list', '--format=json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                accounts = json.loads(result.stdout)
                is_active = len(accounts) > 0 and any(a.get('status') == 'ACTIVE' for a in accounts)
                logging.debug(f"gcloud auth check: {is_active} (Accounts: {len(accounts)})")
                return is_active
            else:
                logging.warning(f"gcloud auth list failed with code {result.returncode}: {result.stderr}")
        except Exception as e:
            logging.error(f"Error checking gcloud auth: {e}")
        return False
    
    def _run_gcloud_login(self) -> bool:
        """Run gcloud auth login interactively."""
        logging.info("Running gcloud auth login...")
        try:
            # Use application-default credentials for API access
            result = subprocess.run(
                ['gcloud', 'auth', 'application-default', 'login', '--no-launch-browser'],
                timeout=300
            )
            return result.returncode == 0
        except Exception as e:
            logging.error(f"gcloud auth login failed: {e}")
            return False
    
    def _get_token_from_gcloud(self) -> Optional[str]:
        """Get access token from gcloud."""
        retries = 3
        for i in range(retries):
            try:
                logging.debug(f"Getting token from gcloud (attempt {i+1}/{retries})...")
                result = subprocess.run(
                    ['gcloud', 'auth', 'print-access-token', '--quiet'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    token = result.stdout.strip()
                    if token:
                        # Tokens typically last 1 hour
                        self._access_token = token
                        self._token_expiry = time.time() + 3600
                        self._save_token_cache()
                        logging.info("Successfully refreshed access token from gcloud")
                        return token
                else:
                    logging.warning(f"gcloud print-access-token failed (attempt {i+1}): {result.stderr}")
            except Exception as e:
                logging.error(f"Failed to get token from gcloud (attempt {i+1}): {e}")
            
            if i < retries - 1:
                time.sleep(2)
        return None
    
    def get_access_token(self) -> Optional[str]:
        """
        Get a valid OAuth 2.0 access token.
        
        Returns:
            Access token string, or None if unavailable.
        """
        # Check if cached token is still valid
        if self._access_token and self._token_expiry:
            if self._token_expiry > time.time() + TOKEN_EXPIRY_BUFFER:
                return self._access_token
        
        
        # Try to get fresh token from gcloud
        logging.debug("Attempting to get fresh token from gcloud...")
        if self._check_gcloud_installed():
            if self._check_gcloud_authenticated():
                token = self._get_token_from_gcloud()
                if token:
                    return token
                else:
                    logging.warning("Failed to obtain token from gcloud despite being authenticated.")
            else:
                logging.warning("gcloud not authenticated. Run 'gcloud auth login' manually.")
        else:
            logging.warning("gcloud CLI not installed. Jules API will be unavailable.")
        
        return None
    
    def ensure_authenticated(self) -> Tuple[bool, str]:
        """
        Ensure we have valid authentication for Jules API.
        
        Returns:
            Tuple of (success, message)
        """
        # Check for existing valid token
        if self.get_access_token():
            return True, "OAuth token is valid."
        
        # Check if gcloud is installed
        if not self._check_gcloud_installed():
            # Try to install it
            if not self._install_gcloud():
                return False, "gcloud CLI not installed and auto-install failed. Install manually: https://cloud.google.com/sdk/docs/install"
        
        # Check if authenticated
        if not self._check_gcloud_authenticated():
            return False, "gcloud not authenticated. Please run: gcloud auth login"
        
        # Try to get token
        token = self._get_token_from_gcloud()
        if token:
            return True, "OAuth token obtained successfully."
        
        return False, "Failed to obtain OAuth token."
    
    def get_auth_headers(self) -> dict:
        """
        Get HTTP headers for authenticated API requests.
        
        Returns:
            Dict with Authorization header if token available, empty dict otherwise.
        """
        token = self.get_access_token()
        if token:
            return {
                "Authorization": f"Bearer {token}",
                "X-Goog-User-Project": QUOTA_PROJECT  # Required for user credentials
            }
        return {}


# Global singleton instance
_auth_manager: Optional[GoogleAuthManager] = None


def get_auth_manager() -> GoogleAuthManager:
    """Get the global auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = GoogleAuthManager()
    return _auth_manager


def get_jules_access_token() -> Optional[str]:
    """
    Get a valid access token for Jules API.
    
    This is the main function to call from other modules.
    Falls back to JULES_API_KEY env var if OAuth unavailable.
    
    Returns:
        Access token string, or None if unavailable.
    """
    # First try OAuth
    manager = get_auth_manager()
    token = manager.get_access_token()
    if token:
        return token
    
    # Fall back to env var (might be a manually set token)
    env_key = os.environ.get("JULES_API_KEY")
    if env_key:
        logging.warning("Using JULES_API_KEY from environment. This may not work if it's an API key rather than an OAuth token.")
        return env_key
    
    return None


def initialize_google_auth() -> Tuple[bool, str]:
    """
    Initialize Google authentication on startup.
    
    Call this during L.O.V.E. startup to ensure Jules API is available.
    
    Returns:
        Tuple of (success, message)
    """
    manager = get_auth_manager()
    success, message = manager.ensure_authenticated()
    
    if success:
        logging.info(f"Google Auth: {message}")
    else:
        logging.warning(f"Google Auth: {message}")
    
    return success, message


def get_jules_auth_headers() -> dict:
    """
    Get complete HTTP headers for Jules API requests.
    
    Includes Authorization bearer token and quota project header.
    Falls back to X-Goog-Api-Key if OAuth is unavailable.
    
    Returns:
        Dict with auth headers, or empty dict if unavailable.
    """
    manager = get_auth_manager()
    headers = manager.get_auth_headers()
    
    if headers:
        return headers
    
    # Fallback to API Key if available (matches logic in get_jules_access_token)
    env_key = os.environ.get("JULES_API_KEY")
    if env_key:
        logging.debug("OAuth token unavailable, falling back to JULES_API_KEY")
        return {
            "X-Goog-Api-Key": env_key,
            # Note: Quota Project might still be needed or might cause issues depending on API Key config
            # Safe to assume standard API Key usage doesn't need it or will complain if missing
        }
        
    return {}
