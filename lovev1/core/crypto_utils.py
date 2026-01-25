
import os
import json
import base64
from typing import Dict, Any, Union

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


def is_valid_xmr_private_key(private_key: str) -> bool:
    """
    Validates the format of a Monero private key.
    """
    return len(private_key) == 64 and all(c in '0123456789abcdefABCDEF' for c in private_key)


def generate_key() -> bytes:
    """Generates a new encryption key."""
    if CRYPTO_AVAILABLE:
        return Fernet.generate_key()
    else:
        # Fallback key (not secure, just obfuscation)
        return b"obfuscation_key_base64_encoded_dummy_val="


def get_cipher(key: bytes):
    """Returns a cipher suite or None."""
    if CRYPTO_AVAILABLE:
        try:
            return Fernet(key)
        except Exception:
            return None
    return None


def encrypt_data(data: Union[Dict, str, bytes], key: bytes) -> bytes:
    """
    Encrypts data (handles dict/json automatically).
    Falls back to Base64 obfuscation if cryptography lib missing.
    """
    # Convert dict to json string
    if isinstance(data, (dict, list)):
        payload = json.dumps(data).encode("utf-8")
    elif isinstance(data, str):
        payload = data.encode("utf-8")
    else:
        payload = data

    cipher = get_cipher(key)
    if cipher:
        return cipher.encrypt(payload)
    else:
        # Fallback: Simple XOR-like or just Base64 (Obfuscation)
        # Using Base64 reverse for mild obfuscation
        return base64.b64encode(payload)[::-1]


def decrypt_data(encrypted: bytes, key: bytes) -> Any:
    """
    Decrypts data. Returns dict if it was a json object.
    """
    cipher = get_cipher(key)
    
    if cipher:
        try:
            decrypted = cipher.decrypt(encrypted)
        except Exception:
            # Maybe it was stored with fallback?
            try:
                decrypted = base64.b64decode(encrypted[::-1])
            except:
                raise ValueError("Decryption failed")
    else:
        # Fallback decryption
        decrypted = base64.b64decode(encrypted[::-1])

    # Try to parse as JSON
    try:
        return json.loads(decrypted.decode("utf-8"))
    except:
        # Return raw string/bytes
        try:
            return decrypted.decode("utf-8")
        except:
            return decrypted

