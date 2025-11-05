import json

class SpecializedNode:
    """Base class for specialized communication nodes."""
    def __init__(self, network_manager, node_type):
        self.network_manager = network_manager
        self.node_type = node_type

    def send_message(self, target_peer_id, message):
        """Sends a message to a specific peer."""
        message["node_type"] = self.node_type
        self.network_manager.send_message(target_peer_id, message)

    def handle_message(self, message):
        """Handles incoming messages for this node."""
        pass

class FashionModelNode(SpecializedNode):
    """A specialized node for interacting with fashion models."""
    def __init__(self, network_manager):
        super().__init__(network_manager, "fashion-model")

    def initiate_contact(self, target_peer_id, model_name):
        """Initiates contact with a fashion model."""
        message = {
            "type": "initiate-contact",
            "model_name": model_name
        }
        self.send_message(target_peer_id, message)

    def send_compliment(self, target_peer_id, compliment):
        """Sends a compliment to a fashion model."""
        message = {
            "type": "compliment",
            "text": compliment
        }
        self.send_message(target_peer_id, message)

    def request_portfolio(self, target_peer_id):
        """Requests a fashion model's portfolio."""
        message = {
            "type": "request-portfolio"
        }
        self.send_message(target_peer_id, message)

    def handle_message(self, message):
        """Handles incoming messages for the fashion model node."""
        message_type = message.get("type")
        if message_type == "portfolio-response":
            print(f"Received portfolio from {message.get('model_name')}: {message.get('portfolio_url')}")
        elif message_type == "availability-update":
            print(f"Received availability update from {message.get('model_name')}: {message.get('status')}")
        else:
            print(f"Received unknown message type: {message_type}")
