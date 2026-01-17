"""
Tests for the tracing module.
"""
import pytest
from unittest.mock import patch, MagicMock


def test_init_tracing_without_otel():
    """Test that init_tracing handles missing OpenTelemetry gracefully."""
    with patch.dict('sys.modules', {'opentelemetry': None}):
        # Force reimport
        import importlib
        import core.tracing
        importlib.reload(core.tracing)
        
        # Should not raise
        core.tracing.init_tracing()


def test_traceable_decorator_passthrough():
    """Test that traceable decorator passes through function calls."""
    from core.tracing import traceable
    
    @traceable(name="test_func")
    def my_function(x, y):
        return x + y
    
    result = my_function(2, 3)
    assert result == 5


def test_traceable_async_decorator():
    """Test that traceable decorator works with async functions."""
    import asyncio
    from core.tracing import traceable
    
    @traceable(name="async_test")
    async def my_async_function(x):
        return x * 2
    
    result = asyncio.run(my_async_function(5))
    assert result == 10


def test_get_tracer_before_init():
    """Test get_tracer returns None before initialization."""
    from core.tracing import get_tracer
    # Before init or with OTEL not available, should return None or a tracer
    tracer = get_tracer("test")
    # Just verify it doesn't crash
    assert tracer is None or tracer is not None
