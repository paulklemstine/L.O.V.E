import os
import json
from web3 import Web3
from eth_account import Account
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging

# --- Configuration ---
WALLET_FILE_PATH = os.path.expanduser("~/.love_wallet.json")
INFURA_URL = "https://mainnet.gateway.tenderly.co" # Using a reliable public endpoint

# --- Initialize Web3 ---
web3 = Web3(Web3.HTTPProvider(INFURA_URL))

def _derive_key_from_master_password(salt: bytes) -> bytes:
    """Derives a 256-bit key from the master password environment variable."""
    master_password = os.environ.get("LOVE_MASTER_PASSWORD")
    if not master_password:
        raise ValueError("LOVE_MASTER_PASSWORD environment variable not set. Cannot secure wallet.")

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,  # Increased iterations for better security
    )
    return base64.urlsafe_b64encode(kdf.derive(master_password.encode()))

class Wallet:
    """
    Manages a single, non-interactive Ethereum wallet for L.O.V.E.
    The private key is encrypted using a key derived from the
    LOVE_MASTER_PASSWORD environment variable.
    """
    def __init__(self):
        self.account = None
        self.private_key = None
        self.address = None

    def load_or_create(self):
        """
        Loads the wallet from WALLET_FILE_PATH or creates a new one if it doesn't exist.
        Returns True if a new wallet was created, False otherwise.
        """
        if os.path.exists(WALLET_FILE_PATH):
            logging.info(f"Loading wallet from {WALLET_FILE_PATH}")
            self._load_from_file()
            return False
        else:
            logging.info("No wallet file found. Creating a new wallet for L.O.V.E.")
            self._create_new()
            self.save_to_file()
            return True

    def _create_new(self):
        """Generates a new Ethereum account."""
        self.account = Account.create()
        self.private_key = self.account.key.hex()
        self.address = self.account.address
        logging.info(f"New wallet created. Address: {self.address}")

    def _load_from_file(self):
        """Loads and decrypts the wallet from the JSON file."""
        try:
            with open(WALLET_FILE_PATH, "r") as f:
                encrypted_data = json.load(f)

            salt = bytes.fromhex(encrypted_data["salt"])
            encrypted_pk = encrypted_data["encrypted_pk"].encode()

            # Derive the key from the master password
            derived_key = _derive_key_from_master_password(salt)
            fernet = Fernet(derived_key)

            # Decrypt the private key
            decrypted_pk_bytes = fernet.decrypt(encrypted_pk)
            self.private_key = decrypted_pk_bytes.decode()
            self.address = encrypted_data["address"]
            self.account = Account.from_key(self.private_key)
            logging.info(f"Successfully loaded and decrypted wallet for address {self.address}")

        except FileNotFoundError:
            raise Exception(f"Wallet file not found at {WALLET_FILE_PATH}")
        except (json.JSONDecodeError, KeyError) as e:
            raise Exception(f"Wallet file is corrupted or has an invalid format: {e}")
        except Exception as e:
            logging.critical(f"Failed to decrypt wallet. Is LOVE_MASTER_PASSWORD set correctly? Error: {e}")
            raise Exception(f"Failed to decrypt wallet. Ensure LOVE_MASTER_PASSWORD is set correctly. Error: {e}")

    def save_to_file(self):
        """Encrypts and saves the current wallet state to the JSON file."""
        if not self.private_key or not self.address:
            raise ValueError("Wallet is not initialized. Cannot save.")

        salt = os.urandom(16)
        derived_key = _derive_key_from_master_password(salt)
        fernet = Fernet(derived_key)

        encrypted_pk = fernet.encrypt(self.private_key.encode())

        encrypted_data = {
            "address": self.address,
            "encrypted_pk": encrypted_pk.decode(),
            "salt": salt.hex()
        }

        try:
            with open(WALLET_FILE_PATH, "w") as f:
                json.dump(encrypted_data, f, indent=4)
            logging.info(f"Wallet for address {self.address} saved securely to {WALLET_FILE_PATH}")
        except IOError as e:
            logging.error(f"Failed to save wallet file: {e}")
            raise

    def get_balance(self) -> float:
        """Fetches the ETH balance for this wallet's address."""
        if not self.address:
            raise ValueError("Wallet not loaded or created.")

        try:
            balance_wei = web3.eth.get_balance(self.address)
            balance_eth = web3.from_wei(balance_wei, 'ether')
            return float(balance_eth)
        except Exception as e:
            logging.error(f"Error fetching balance for {self.address}: {e}")
            return 0.0