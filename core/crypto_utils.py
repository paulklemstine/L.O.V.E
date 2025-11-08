def is_valid_xmr_private_key(private_key: str) -> bool:
    """
    Validates the format of a Monero private key.
    """
    return len(private_key) == 64 and all(c in '0123456789abcdefABCDEF' for c in private_key)
