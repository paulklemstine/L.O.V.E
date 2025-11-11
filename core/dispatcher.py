from typing import Any, Dict

async def dispatch_structured_payload(payload: Dict[str, Any], interface_handler: Any) -> Dict[str, Any]:
    """
    Dispatches a structured data payload to an external service interface.

    Args:
        payload: A dictionary representing the data payload.
        interface_handler: An object responsible for processing interactions with a specific external service.

    Returns:
        A status dictionary indicating the outcome of the dispatch operation.
    """
    return await interface_handler.handle_payload(payload)
