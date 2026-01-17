"""generated_code.py – Fixed version that safely accesses nested data.

The original implementation raised

    ``TypeError: string indices must be integers, not 'str'``

when it tried to index a string with a non‑integer key.  This version
introduces ``safe_get`` helpers that validate the container type before
indexing, ensuring that only appropriate keys/indices are used.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Mapping, Sequence


log = logging.getLogger(__name__)


def safe_get(container: Any, key: Any) -> Any:
    """
    Retrieve ``key`` from ``container`` with strict type checking.

    - If ``container`` is a mapping (e.g. ``dict``), ``key`` is used as a
      dictionary key.
    - If ``container`` is a sequence (``list``, ``tuple``) but *not* a
      string/bytes, ``key`` must be an ``int``.
    - For any other combination a descriptive ``TypeError`` is raised.

    The function propagates ``KeyError`` when a mapping key is missing,
    allowing callers to decide how to handle it.

    Parameters
    ----------
    container:
        The object to index.
    key:
        The key or index to retrieve.

    Returns
    -------
    Any
        The value stored at ``container[key]``.

    Raises
    ------
    TypeError
        If ``container`` cannot be indexed with ``key`` (wrong type).
    KeyError
        If ``container`` is a mapping and ``key`` does not exist.
    """
    if isinstance(container, Mapping):
        # Mapping access – let KeyError bubble up if the key is absent.
        return container[key]

    if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        if not isinstance(key, int):
            raise TypeError(
                f"Sequence indexing requires an int index, got {type(key).__name__}"
            )
        return container[key]

    raise TypeError(
        f"Object of type {type(container).__name__} cannot be indexed with {type(key).__name__}"
    )


def process_event(event: dict) -> dict:
    """
    Safely extract a nested value from an incoming event dictionary.

    The original buggy code performed something akin to:

        items = event['payload']['items'][0]

    which failed when ``event['payload']['items']`` was a string instead of a
    list, yielding the ``string indices must be integers, not 'str'`` error.

    This implementation uses :func:`safe_get` to validate each level before
    indexing, thereby preventing the exception.

    Example input (JSON structure):
        {
            "payload": {
                "items": [
                    {"value": 42}
                ]
            }
        }

    Returns
    -------
    dict
        A dictionary containing the processed result, e.g.
        ``{"processed_value": 42}``.
    """
    # 1️⃣ Safely obtain the top‑level ``payload`` dictionary.
    payload = safe_get(event, "payload")  # expects a Mapping

    # 2️⃣ Safely obtain the ``items`` collection; it should be a Sequence.
    items = safe_get(payload, "items")

    # 3️⃣ Safely fetch the first element using an integer index.
    first_item = safe_get(items, 0)  # ``key`` must be int

    # 4️⃣ Retrieve the final value (e.g. a field called ``value``).
    result_value = safe_get(first_item, "value")

    return {"processed_value": result_value}


def main() -> None:
    """
    Script entry point.

    Reads a JSON document from *stdin*, processes it via :func:`process_event`,
    and prints the resulting dictionary to *stdout*.

    Logging is configured to ``INFO`` level for normal operation; set the
    level to ``DEBUG`` if more verbosity is required during development.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stderr,
    )

    try:
        raw_input = sys.stdin.read()
        event_data = json.loads(raw_input)
    except json.JSONDecodeError as exc:
        log.error("Failed to parse JSON input: %s", exc)
        sys.exit(1)

    try:
        result = process_event(event_data)  # type: ignore[arg-type]
        print(json.dumps(result))
    except (KeyError, TypeError) as exc:
        log.exception("Error while processing event: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
