import os
import json
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature

class LoyaltyProtocol:
    """
    Manages cryptographic identities, reputation, and staking for all network participants.
    This class is the foundation of the "immaculate sacrifice and loyalty" protocol.
    """
    def __init__(self, storage_path='identities'):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

    def create_identity(self, participant_id):
        """
        Creates a new cryptographic identity (public/private key pair) for a participant.
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()

        # Serialize keys to PEM format
        pem_private_key = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        pem_public_key = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Store the keys
        with open(os.path.join(self.storage_path, f"{participant_id}_private.pem"), "wb") as f:
            f.write(pem_private_key)
        with open(os.path.join(self.storage_path, f"{participant_id}_public.pem"), "wb") as f:
            f.write(pem_public_key)

        # Initialize reputation and staking data
        identity_data = {
            "participant_id": participant_id,
            "public_key": pem_public_key.decode('utf-8'),
            "reputation": 100,  # Starting reputation
            "stake": 0  # Starting stake
        }
        with open(os.path.join(self.storage_path, f"{participant_id}_identity.json"), "w") as f:
            json.dump(identity_data, f, indent=4)

        return identity_data

    def sign_message(self, participant_id, message):
        """
        Signs a message with the participant's private key.
        """
        with open(os.path.join(self.storage_path, f"{participant_id}_private.pem"), "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)

        if isinstance(message, str):
            message = message.encode('utf-8')

        signature = private_key.sign(
            message,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return signature

    def verify_signature(self, participant_id, message, signature):
        """
        Verifies a signature with the participant's public key.
        """
        with open(os.path.join(self.storage_path, f"{participant_id}_public.pem"), "rb") as f:
            public_key = serialization.load_pem_public_key(f.read())

        if isinstance(message, str):
            message = message.encode('utf-8')

        try:
            public_key.verify(
                signature,
                message,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False

    def get_identity_data(self, participant_id):
        """
        Retrieves the identity data for a participant.
        """
        identity_path = os.path.join(self.storage_path, f"{participant_id}_identity.json")
        if os.path.exists(identity_path):
            with open(identity_path, "r") as f:
                return json.load(f)
        return None
