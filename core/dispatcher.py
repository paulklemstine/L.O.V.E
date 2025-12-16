import asyncio
from typing import Any, Dict, List, Callable, Awaitable
from dataclasses import dataclass, field
import uuid
from datetime import datetime
import logging

@dataclass
class Event:
    """
    Represents a discrete event in the system.
    """
    type: str  # e.g., "TASK_POSTED", "ACTION_REQUIRED"
    payload: Dict[str, Any]
    source: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

class EventDispatcher:
    """
    A central message bus for the Blackboard architecture.
    Allows agents and components to subscribe to specific event types and
    broadcast events to subscribers.
    """
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], Awaitable[None]]]] = {}
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._logger = logging.getLogger("EventDispatcher")

    def subscribe(self, event_type: str, callback: Callable[[Event], Awaitable[None]]):
        """
        Subscribes a callback function to a specific event type.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        self._logger.debug(f"Subscribed to {event_type}")

    async def publish(self, event: Event):
        """
        Publishes an event to the bus. The event is added to a queue for processing.
        """
        self._logger.info(f"Event Published: {event.type} from {event.source}")
        await self._event_queue.put(event)

    async def process_events(self):
        """
        Processes events in the queue until it is empty.
        This is typically called efficiently within the main loop or a router node.
        """
        while not self._event_queue.empty():
            event = await self._event_queue.get()
            await self._dispatch_event(event)

    async def _dispatch_event(self, event: Event):
        """
        Dispatches a single event to all its subscribers.
        """
        if event.type in self._subscribers:
            subscribers = self._subscribers[event.type]
            # Execute all callbacks concurrently
            tasks = [callback(event) for callback in subscribers]
            if tasks:
                await asyncio.gather(*tasks)
        else:
            self._logger.debug(f"No subscribers for event type: {event.type}")

    # Legacy method for backward compatibility if needed, though we plan to deprecate it.
    async def dispatch_structured_payload(self, payload: Dict[str, Any], interface_handler: Any) -> Dict[str, Any]:
         return await interface_handler.handle_payload(payload)

# Global singleton instance if needed, or instantiate per graph
global_dispatcher = EventDispatcher()
