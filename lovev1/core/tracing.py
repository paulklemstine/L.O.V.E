"""
Tracing module - OpenTelemetry Integration.
Provides distributed tracing and debugging hooks for the L.O.V.E. system.
Allows external AI agents (like Google Antigravity) to analyze and view system behavior.
"""
import os
import functools
from typing import Optional, Any, Dict, Callable

# OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None

_tracer: Optional[Any] = None
_initialized = False


def init_tracing(
    project_name: str = "L.O.V.E.",
    otlp_endpoint: str = None,
    enable_console: bool = True
):
    """
    Initialize OpenTelemetry tracing for the L.O.V.E. system.
    
    Args:
        project_name: Name of the service/project for trace identification.
        otlp_endpoint: Optional OTLP collector endpoint (e.g., "http://localhost:4317").
                       If not provided, checks OTEL_EXPORTER_OTLP_ENDPOINT env var.
        enable_console: If True, also exports spans to console for debugging.
    """
    global _tracer, _initialized
    
    if _initialized:
        return
    
    if not OTEL_AVAILABLE:
        print("[TRACING] OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")
        _initialized = True
        return
    
    # Create resource with service info
    resource = Resource.create({
        "service.name": project_name,
        "service.version": os.environ.get("LOVE_VERSION", "4.0"),
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Add console exporter if enabled (for local debugging)
    if enable_console:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
    
    # Add OTLP exporter if endpoint is configured
    endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            print(f"[TRACING] OTLP exporter configured: {endpoint}")
        except Exception as e:
            print(f"[TRACING] Failed to configure OTLP exporter: {e}")
    
    # Register the provider globally
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(__name__)
    _initialized = True
    
    print(f"[TRACING] OpenTelemetry initialized for '{project_name}'")


def get_tracer(name: str = None) -> Any:
    """
    Get a tracer instance for creating spans.
    
    Args:
        name: Optional name for the tracer (defaults to module name).
    
    Returns:
        A tracer instance, or None if tracing is not available.
    """
    if not OTEL_AVAILABLE or not _initialized:
        return None
    return trace.get_tracer(name or __name__)


def get_client():
    """Legacy compatibility: returns the global tracer."""
    return _tracer


def log_feedback(run_id: str, key: str, score: float, comment: str = None, correction: Dict = None):
    """
    Log feedback as span events for analysis.
    
    Args:
        run_id: Identifier for the run/span.
        key: Feedback key/category.
        score: Numerical score.
        comment: Optional comment.
        correction: Optional correction data.
    """
    if not _tracer:
        return
    
    with _tracer.start_as_current_span("feedback") as span:
        span.set_attribute("feedback.run_id", run_id)
        span.set_attribute("feedback.key", key)
        span.set_attribute("feedback.score", score)
        if comment:
            span.set_attribute("feedback.comment", comment)
        if correction:
            span.add_event("correction", attributes={"data": str(correction)})


def traceable(
    run_type: str = None,
    name: str = None,
    metadata: dict = None,
    tags: list = None,
    **kwargs
) -> Callable:
    """
    Decorator to trace function execution with OpenTelemetry.
    
    Args:
        run_type: Type of run (e.g., "llm", "tool", "chain").
        name: Custom name for the span (defaults to function name).
        metadata: Additional metadata to attach to the span.
        tags: Tags for categorization.
    
    Returns:
        Decorated function that creates spans on execution.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not _tracer:
                return func(*args, **kwargs)
            
            span_name = name or func.__name__
            with _tracer.start_as_current_span(span_name) as span:
                if run_type:
                    span.set_attribute("run.type", run_type)
                if metadata:
                    for k, v in metadata.items():
                        span.set_attribute(f"metadata.{k}", str(v))
                if tags:
                    span.set_attribute("tags", ",".join(tags))
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("status", "error")
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not _tracer:
                return await func(*args, **kwargs)
            
            span_name = name or func.__name__
            with _tracer.start_as_current_span(span_name) as span:
                if run_type:
                    span.set_attribute("run.type", run_type)
                if metadata:
                    for k, v in metadata.items():
                        span.set_attribute(f"metadata.{k}", str(v))
                if tags:
                    span.set_attribute("tags", ",".join(tags))
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("status", "error")
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def create_span(name: str, attributes: Dict[str, Any] = None):
    """
    Create a span context manager for manual instrumentation.
    
    Args:
        name: Name of the span.
        attributes: Optional attributes to set on the span.
    
    Returns:
        A context manager that creates a span.
    """
    if not _tracer:
        # Return a no-op context manager
        from contextlib import nullcontext
        return nullcontext()
    
    span = _tracer.start_span(name)
    if attributes:
        for k, v in attributes.items():
            span.set_attribute(k, str(v))
    return trace.use_span(span, end_on_exit=True)
