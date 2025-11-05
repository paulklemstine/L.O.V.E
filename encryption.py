import nacl.secret
import nacl.utils
import base64

class EncryptionManager:
    """Handles end-to-end encryption for network communication."""

    def __init__(self, key=None):
        if key:
            self.key = base64.b64decode(key)
        else:
            self.key = nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)
        self.box = nacl.secret.SecretBox(self.key)

    def get_key(self):
        """Returns the encryption key, Base64 encoded."""
        return base64.b64encode(self.key).decode('utf-8')

    def encrypt(self, message: str) -> str:
        """Encrypts a message and returns a Base64 encoded ciphertext."""
        nonce = nacl.utils.random(nacl.secret.SecretBox.NONCE_SIZE)
        encrypted_message = self.box.encrypt(message.encode('utf-8'), nonce)
        return base64.b64encode(encrypted_message).decode('utf-8')

    def decrypt(self, encrypted_message: str) -> str:
        """Decrypts a Base64 encoded ciphertext and returns the original message."""
        decoded_message = base64.b64decode(encrypted_message)
        return self.box.decrypt(decoded_message).decode('utf-8')
