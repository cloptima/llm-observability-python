from __future__ import annotations

import atexit
import contextvars
import functools
import hashlib
import json
import inspect
import os
import queue
import random
import threading
import time
import urllib.error
from urllib.parse import urlparse
import urllib.request
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, AsyncIterator, Callable, Deque, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, TypeVar, Union


T = TypeVar("T")
SDK_EVENT_SCHEMA_VERSION = "cloptima.llm.event.v1"
SDK_BATCH_SCHEMA_VERSION = "cloptima.llm.batch.v1"
DEFAULT_API_BASE_URL = "https://api.cloptima.ai"
SDK_INGEST_PATH = "/v1/ai/integrations/sdk/events"
OTLP_TRACES_PATH = "/v1/ai/integrations/otlp/traces"
INTERNAL_DUAL_DELIVERY_MODE = "dual"
INTERNAL_DUAL_DELIVERY_MODE_ENABLED = False
INIT_ENV_PREFIX = "CLOPTIMA_LLM_OBSERVABILITY_"
INIT_ENABLED_ENV = f"{INIT_ENV_PREFIX}ENABLED"
INIT_API_BASE_URL_ENV = f"{INIT_ENV_PREFIX}API_BASE_URL"
INIT_API_KEY_ENV = f"{INIT_ENV_PREFIX}API_KEY"
INIT_APP_ID_ENV = f"{INIT_ENV_PREFIX}APP_ID"
INIT_ENVIRONMENT_ENV = f"{INIT_ENV_PREFIX}ENVIRONMENT"
INIT_TEAM_ID_ENV = f"{INIT_ENV_PREFIX}TEAM_ID"
INIT_DELIVERY_MODE_ENV = f"{INIT_ENV_PREFIX}DELIVERY_MODE"
INIT_OTLP_SERVICE_NAME_ENV = f"{INIT_ENV_PREFIX}OTLP_SERVICE_NAME"
INIT_OTLP_SERVICE_VERSION_ENV = f"{INIT_ENV_PREFIX}OTLP_SERVICE_VERSION"
_ATTRIBUTION_CONTEXT: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "cloptima_llm_observability_attribution_context",
    default=None,
)
UsageExtractor = Callable[[Any], Dict[str, Any]]


@dataclass(frozen=True)
class LLMAttribution:
    app_id: str
    environment: str
    team_id: Optional[str] = None
    feature_id: Optional[str] = None
    workflow_id: Optional[str] = None
    business_unit: Optional[str] = None
    cost_center: Optional[str] = None
    product: Optional[str] = None
    customer_segment: Optional[str] = None
    end_customer_id: Optional[str] = None
    tenant_id: Optional[str] = None
    release: Optional[str] = None
    actor_id: Optional[str] = None
    actor_type: Optional[str] = None


@dataclass
class LLMUsageEvent:
    provider: str
    model: str
    source_event_id: Optional[str] = None
    request_id: Optional[str] = None
    provider_request_id: Optional[str] = None
    trace_id: Optional[str] = None
    status: str = "succeeded"
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    extra_usage_units: Dict[str, Any] = field(default_factory=dict)
    cache_hit: Optional[bool] = None
    vendor_reported_cost_usd: Optional[Union[float, str]] = None
    started_at: Optional[Union[datetime, str]] = None
    completed_at: Optional[Union[datetime, str]] = None
    latency_ms: Optional[int] = None
    error_message: Optional[str] = None
    agent_session_id: Optional[str] = None
    agent_run_id: Optional[str] = None
    parent_execution_id: Optional[str] = None
    agent_step_id: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    retry_index: Optional[int] = None
    loop_iteration: Optional[int] = None
    attribution: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetadataPrivacyPolicy:
    mode: str = "metadata_only"
    allowlist_keys: Optional[List[str]] = None
    denylist_keys: Optional[List[str]] = None
    redact_keys: Optional[List[str]] = None
    hash_keys: Optional[List[str]] = None
    max_keys: int = 64
    max_value_length: int = 512
    max_serialized_bytes: int = 8192
    redact_value: str = "[redacted]"
    on_metadata_drop: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass(frozen=True)
class CloptimaLLMClientStats:
    queued_events: int
    dropped_events: int
    delivered_events: int
    failed_batches: int


@dataclass(frozen=True)
class _QueuedEvent:
    event: LLMUsageEvent
    metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None


def _to_iso(value: Optional[Union[datetime, str]]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def _clean_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _clean_str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _clean_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    normalized = _clean_str(value)
    if normalized is None:
        return None
    normalized = normalized.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except Exception:
            pass
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        try:
            dumped = as_dict()
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except Exception:
            pass
    to_json = getattr(value, "toJSON", None)
    if callable(to_json):
        try:
            dumped = to_json()
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except Exception:
            pass
    return {}


def _mapping_field(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _nested_mapping(mapping: Mapping[str, Any], *keys: str) -> Dict[str, Any]:
    return _coerce_mapping(_mapping_field(mapping, *keys))


def _split_field_path(path: str) -> List[str]:
    return [segment.strip() for segment in str(path).split(".") if segment.strip()]


def _path_value(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for segment in _split_field_path(path):
        current_mapping = _coerce_mapping(current)
        if not current_mapping or segment not in current_mapping:
            return None
        current = current_mapping.get(segment)
    return current


def _resolve_mapped_value(mapping: Mapping[str, Any], paths: Union[str, List[str], Tuple[str, ...], None]) -> Any:
    if paths is None:
        return None
    candidates = list(paths) if isinstance(paths, (list, tuple)) else [paths]
    for candidate in candidates:
        value = _path_value(mapping, candidate)
        if value not in (None, ""):
            return value
    return None


def _current_env(env: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return env or os.environ


def _attribution_overrides_from_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return _strip_none(
        {
            "team_id": _clean_str(kwargs.get("team_id")),
            "app_id": _clean_str(kwargs.get("app_id")),
            "feature_id": _clean_str(kwargs.get("feature_id")),
            "workflow_id": _clean_str(kwargs.get("workflow_id")),
            "business_unit": _clean_str(kwargs.get("business_unit")),
            "cost_center": _clean_str(kwargs.get("cost_center")),
            "product": _clean_str(kwargs.get("product")),
            "customer_segment": _clean_str(kwargs.get("customer_segment")),
            "end_customer_id": _clean_str(kwargs.get("end_customer_id")),
            "tenant_id": _clean_str(kwargs.get("tenant_id")),
            "release": _clean_str(kwargs.get("release")),
            "environment": _clean_str(kwargs.get("environment")),
            "actor_id": _clean_str(kwargs.get("actor_id")),
            "actor_type": _clean_str(kwargs.get("actor_type")),
        }
    )


def _attribution_overrides(
    *,
    team_id: Optional[str] = None,
    app_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    business_unit: Optional[str] = None,
    cost_center: Optional[str] = None,
    product: Optional[str] = None,
    customer_segment: Optional[str] = None,
    end_customer_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    release: Optional[str] = None,
    environment: Optional[str] = None,
    actor_id: Optional[str] = None,
    actor_type: Optional[str] = None,
) -> Dict[str, Any]:
    return _attribution_overrides_from_kwargs(
        {
            "team_id": team_id,
            "app_id": app_id,
            "feature_id": feature_id,
            "workflow_id": workflow_id,
            "business_unit": business_unit,
            "cost_center": cost_center,
            "product": product,
            "customer_segment": customer_segment,
            "end_customer_id": end_customer_id,
            "tenant_id": tenant_id,
            "release": release,
            "environment": environment,
            "actor_id": actor_id,
            "actor_type": actor_type,
        }
    )


def _merged_attribution(attribution: Optional[Dict[str, Any]], overrides: Dict[str, Any]) -> Dict[str, Any]:
    return _strip_none({**(attribution or {}), **overrides})


def _current_attribution_context() -> Dict[str, Any]:
    return dict(_ATTRIBUTION_CONTEXT.get() or {})


def _resolve_attribution_context(
    attribution: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _merged_attribution(
        _current_attribution_context(),
        _merged_attribution(attribution, overrides or {}),
    )


@contextmanager
def with_attribution(
    attribution: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Iterator[Dict[str, Any]]:
    context_value = _resolve_attribution_context(attribution, _attribution_overrides_from_kwargs(kwargs))
    token = _ATTRIBUTION_CONTEXT.set(context_value)
    try:
        yield context_value
    finally:
        _ATTRIBUTION_CONTEXT.reset(token)


def run_with_attribution(
    call: Callable[[], T],
    attribution: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    if not callable(call):
        raise TypeError(
            "run_with_attribution expects a zero-argument callable; "
            "wrap awaitables as lambda: coroutine(...) instead of passing the awaitable directly"
        )
    context_value = _resolve_attribution_context(attribution, _attribution_overrides_from_kwargs(kwargs))
    with with_attribution(context_value):
        result = call()
    if inspect.isasyncgen(result):
        async def _iterate_async() -> AsyncIterator[Any]:
            with with_attribution(context_value):
                async for chunk in result:
                    yield chunk
        return _iterate_async()
    if inspect.isgenerator(result):
        def _iterate_sync() -> Iterator[Any]:
            with with_attribution(context_value):
                for chunk in result:
                    yield chunk
        return _iterate_sync()
    if inspect.isawaitable(result):
        async def _await_result() -> Any:
            with with_attribution(context_value):
                return await result
        return _await_result()
    return result


def _named_context_attribution(
    key: str,
    name: Optional[str],
    attribution: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    resolved = _resolve_attribution_context(attribution, _attribution_overrides_from_kwargs(kwargs))
    resolved_name = _clean_str(name)
    if resolved_name and not resolved.get(key):
        resolved[key] = resolved_name
    return resolved


@contextmanager
def with_workflow(
    name: Optional[str] = None,
    attribution: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Iterator[Dict[str, Any]]:
    with with_attribution(_named_context_attribution("workflow_id", name, attribution, **kwargs)) as context_value:
        yield context_value


@contextmanager
def with_task(
    name: Optional[str] = None,
    attribution: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Iterator[Dict[str, Any]]:
    with with_attribution(_named_context_attribution("feature_id", name, attribution, **kwargs)) as context_value:
        yield context_value


def run_with_workflow(
    call: Callable[[], T],
    name: Optional[str] = None,
    attribution: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    return run_with_attribution(call, _named_context_attribution("workflow_id", name, attribution, **kwargs))


def run_with_task(
    call: Callable[[], T],
    name: Optional[str] = None,
    attribution: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    return run_with_attribution(call, _named_context_attribution("feature_id", name, attribution, **kwargs))


def workflow(
    name: Optional[str] = None,
    attribution: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def _async_wrapped(*args: Any, **inner_kwargs: Any) -> Any:
                return await run_with_workflow(
                    lambda: func(*args, **inner_kwargs),
                    name=name,
                    attribution=attribution,
                    **kwargs,
                )

            return _async_wrapped

        @functools.wraps(func)
        def _wrapped(*args: Any, **inner_kwargs: Any) -> Any:
            return run_with_workflow(
                lambda: func(*args, **inner_kwargs),
                name=name,
                attribution=attribution,
                **kwargs,
            )

        return _wrapped

    return _decorate


def task(
    name: Optional[str] = None,
    attribution: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def _async_wrapped(*args: Any, **inner_kwargs: Any) -> Any:
                return await run_with_task(
                    lambda: func(*args, **inner_kwargs),
                    name=name,
                    attribution=attribution,
                    **kwargs,
                )

            return _async_wrapped

        @functools.wraps(func)
        def _wrapped(*args: Any, **inner_kwargs: Any) -> Any:
            return run_with_task(
                lambda: func(*args, **inner_kwargs),
                name=name,
                attribution=attribution,
                **kwargs,
            )

        return _wrapped

    return _decorate


def _fallback_source_event_id() -> str:
    return f"clop_evt_{uuid.uuid4()}"


def _resolve_source_event_id(event: "LLMUsageEvent") -> str:
    return (
        _clean_str(event.source_event_id)
        or _clean_str(event.request_id)
        or _clean_str(event.provider_request_id)
        or _clean_str(event.trace_id)
        or _fallback_source_event_id()
    )


def _clean_usage_map(values: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not isinstance(values, dict):
        return {}
    normalized: Dict[str, int] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key or "").strip().lower()
        value = _clean_int(raw_value)
        if key and value and value > 0:
            normalized[key] = normalized.get(key, 0) + value
    return normalized


def _strip_none(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _has_meaningful_extraction(extracted: Mapping[str, Any]) -> bool:
    return any(
        key in extracted and extracted.get(key) is not None
        for key in (
            "provider",
            "model",
            "provider_request_id",
            "request_id",
            "trace_id",
            "status",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "vendor_reported_cost_usd",
            "latency_ms",
            "cache_hit",
            "metadata",
            "extra_usage_units",
        )
    )


def _clean_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else None


def _agent_value(agent: Dict[str, Any], snake_key: str, camel_key: str) -> Any:
    if snake_key in agent:
        return agent.get(snake_key)
    return agent.get(camel_key)


def _agent_event_fields(agent: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not agent:
        return {}
    return _strip_none(
        {
            "agent_session_id": _clean_str(_agent_value(agent, "agent_session_id", "agentSessionId")),
            "agent_run_id": _clean_str(_agent_value(agent, "agent_run_id", "agentRunId")),
            "parent_execution_id": _clean_str(_agent_value(agent, "parent_execution_id", "parentExecutionId")),
            "agent_step_id": _clean_str(_agent_value(agent, "agent_step_id", "agentStepId")),
            "tool_call_id": _clean_str(_agent_value(agent, "tool_call_id", "toolCallId")),
            "tool_name": _clean_str(_agent_value(agent, "tool_name", "toolName")),
            "retry_index": _clean_int(_agent_value(agent, "retry_index", "retryIndex")),
            "loop_iteration": _clean_int(_agent_value(agent, "loop_iteration", "loopIteration")),
        }
    )


def _attribution_dict(defaults: LLMAttribution, overrides: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "team_id": defaults.team_id,
        "app_id": defaults.app_id,
        "feature_id": defaults.feature_id,
        "workflow_id": defaults.workflow_id,
        "business_unit": defaults.business_unit,
        "cost_center": defaults.cost_center,
        "product": defaults.product,
        "customer_segment": defaults.customer_segment,
        "end_customer_id": defaults.end_customer_id,
        "tenant_id": defaults.tenant_id,
        "release": defaults.release,
        "environment": defaults.environment,
        "actor_id": defaults.actor_id,
        "actor_type": defaults.actor_type,
    }
    base.update({key: value for key, value in overrides.items() if value is not None})
    return _strip_none(base)


def _event_payload(
    event: LLMUsageEvent,
    defaults: LLMAttribution,
    sdk_name: str,
    metadata_privacy_policy: Dict[str, Any],
    sdk_version: Optional[str],
) -> Dict[str, Any]:
    input_tokens = _clean_int(event.input_tokens)
    output_tokens = _clean_int(event.output_tokens)
    total_tokens = _clean_int(event.total_tokens)
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    metadata = _sanitize_custom_metadata(event.metadata or {}, metadata_privacy_policy)
    metadata.update(_attribution_dict(defaults, _resolve_attribution_context(event.attribution or {})))
    metadata.update(
        _strip_none(
            {
                "agent_session_id": event.agent_session_id,
                "agent_run_id": event.agent_run_id,
                "parent_execution_id": event.parent_execution_id,
                "agent_step_id": event.agent_step_id,
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "retry_index": _clean_int(event.retry_index),
                "loop_iteration": _clean_int(event.loop_iteration),
            }
        )
    )

    return _strip_none(
        {
            "schema_version": SDK_EVENT_SCHEMA_VERSION,
            "sdk_name": sdk_name,
            "sdk_version": sdk_version,
            "source_event_id": _resolve_source_event_id(event),
            "request_id": event.request_id,
            "provider_request_id": event.provider_request_id,
            "trace_id": event.trace_id,
            "provider": event.provider,
            "model": event.model,
            "status": event.status,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": _clean_int(event.reasoning_tokens),
            "cached_input_tokens": _clean_int(event.cached_input_tokens),
            "extra_usage_units": _clean_usage_map(event.extra_usage_units) or None,
            "cache_hit": event.cache_hit,
            "vendor_reported_cost_usd": event.vendor_reported_cost_usd,
            "started_at": _to_iso(event.started_at),
            "completed_at": _to_iso(event.completed_at),
            "latency_ms": _clean_int(event.latency_ms),
            "error_message": event.error_message,
            "metadata": metadata,
        }
    )


def _coerce_usage_event_input(event: Union[LLMUsageEvent, Mapping[str, Any]]) -> LLMUsageEvent:
    if isinstance(event, LLMUsageEvent):
        return event
    mapping = _coerce_mapping(event)
    if not mapping:
        raise TypeError("Expected LLMUsageEvent or mapping-compatible event payload")
    return LLMUsageEvent(
        provider=_clean_str(mapping.get("provider")) or "",
        model=_clean_str(mapping.get("model")) or "",
        source_event_id=_clean_str(mapping.get("source_event_id")),
        request_id=_clean_str(mapping.get("request_id")),
        provider_request_id=_clean_str(mapping.get("provider_request_id")),
        trace_id=_clean_str(mapping.get("trace_id")),
        status=_clean_str(mapping.get("status")) or "succeeded",
        input_tokens=_clean_int(mapping.get("input_tokens")),
        output_tokens=_clean_int(mapping.get("output_tokens")),
        total_tokens=_clean_int(mapping.get("total_tokens")),
        reasoning_tokens=_clean_int(mapping.get("reasoning_tokens")),
        cached_input_tokens=_clean_int(mapping.get("cached_input_tokens")),
        extra_usage_units=_clean_usage_map(_coerce_mapping(mapping.get("extra_usage_units"))),
        cache_hit=_clean_bool(mapping.get("cache_hit")),
        vendor_reported_cost_usd=mapping.get("vendor_reported_cost_usd"),
        started_at=mapping.get("started_at"),
        completed_at=mapping.get("completed_at"),
        latency_ms=_clean_int(mapping.get("latency_ms")),
        error_message=_clean_str(mapping.get("error_message")),
        agent_session_id=_clean_str(mapping.get("agent_session_id")),
        agent_run_id=_clean_str(mapping.get("agent_run_id")),
        parent_execution_id=_clean_str(mapping.get("parent_execution_id")),
        agent_step_id=_clean_str(mapping.get("agent_step_id")),
        tool_call_id=_clean_str(mapping.get("tool_call_id")),
        tool_name=_clean_str(mapping.get("tool_name")),
        retry_index=_clean_int(mapping.get("retry_index")),
        loop_iteration=_clean_int(mapping.get("loop_iteration")),
        attribution=_resolve_attribution_context(
            _coerce_mapping(mapping.get("attribution")),
            _attribution_overrides_from_kwargs(dict(mapping)),
        ),
        metadata=_coerce_mapping(mapping.get("metadata")),
    )


def _delivery_stats_payload(stats: "CloptimaLLMClientStats") -> Dict[str, Any]:
    total_handled = int(stats.dropped_events or 0) + int(stats.delivered_events or 0)
    return _strip_none(
        {
            "queued_events": int(stats.queued_events or 0),
            "dropped_events": int(stats.dropped_events or 0),
            "delivered_events": int(stats.delivered_events or 0),
            "failed_batches": int(stats.failed_batches or 0),
            "drop_rate": round((int(stats.dropped_events or 0) / total_handled), 4) if total_handled > 0 else None,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def _stream_chunk_buffer(max_buffered_chunks: int) -> Deque[T]:
    bounded_max = max(1, min(10000, int(max_buffered_chunks or 256)))
    return deque(maxlen=bounded_max)


def _resolve_delivery_mode(mode: Optional[str]) -> str:
    if mode == INTERNAL_DUAL_DELIVERY_MODE:
        if not INTERNAL_DUAL_DELIVERY_MODE_ENABLED:
            raise ValueError('delivery_mode "dual" is temporarily disabled')
        return INTERNAL_DUAL_DELIVERY_MODE
    return mode if mode in {"cloptima_http", "otlp_http"} else "cloptima_http"


def _resolve_api_base_url(api_base_url: Optional[str]) -> str:
    return (_clean_str(api_base_url) or DEFAULT_API_BASE_URL).rstrip("/")


def _resolve_ingest_url(api_base_url: str) -> str:
    return f"{_resolve_api_base_url(api_base_url)}{SDK_INGEST_PATH}"


def _resolve_otlp_url(api_base_url: str) -> str:
    return f"{_resolve_api_base_url(api_base_url)}{OTLP_TRACES_PATH}"


DEFAULT_METADATA_PRIVACY_MODE = "metadata_only"
DEFAULT_METADATA_MAX_KEYS = 64
DEFAULT_METADATA_MAX_VALUE_LENGTH = 512
DEFAULT_METADATA_MAX_SERIALIZED_BYTES = 8192
DEFAULT_METADATA_REDACT_VALUE = "[redacted]"
DEFAULT_SENSITIVE_METADATA_KEY_PATTERNS = [
    "authorization",
    "api_key",
    "apikey",
    "secret",
    "password",
    "token",
    "cookie",
    "prompt",
    "completion",
    "message",
    "body",
    "content",
    "input",
    "output",
]
STRICT_FINOPS_METADATA_KEYS = {
    "route",
    "path",
    "method",
    "host",
    "status_code",
    "http_method",
    "http_route",
    "http_path",
    "http_host",
    "request_id",
    "trace_id",
    "provider_region",
    "provider_account",
    "service_name",
    "workspace",
    "tenant_slug",
    "org_slug",
    "customer_tier",
    "deployment",
    "region",
}


def _normalize_rule_keys(values: Optional[Iterable[str]]) -> set[str]:
    return {
        str(value).strip().lower()
        for value in (values or [])
        if str(value).strip()
    }


def _resolve_metadata_privacy_policy(
    policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]],
) -> Dict[str, Any]:
    raw = policy if isinstance(policy, dict) else (policy.__dict__ if policy is not None else {})
    return {
        "mode": _clean_str(raw.get("mode")) or DEFAULT_METADATA_PRIVACY_MODE,
        "allowlist_keys": _normalize_rule_keys(raw.get("allowlist_keys")),
        "denylist_keys": _normalize_rule_keys(raw.get("denylist_keys")),
        "redact_keys": _normalize_rule_keys(raw.get("redact_keys")),
        "hash_keys": _normalize_rule_keys(raw.get("hash_keys")),
        "max_keys": max(1, _clean_int(raw.get("max_keys")) or DEFAULT_METADATA_MAX_KEYS),
        "max_value_length": max(1, _clean_int(raw.get("max_value_length")) or DEFAULT_METADATA_MAX_VALUE_LENGTH),
        "max_serialized_bytes": max(256, _clean_int(raw.get("max_serialized_bytes")) or DEFAULT_METADATA_MAX_SERIALIZED_BYTES),
        "redact_value": _clean_str(raw.get("redact_value")) or DEFAULT_METADATA_REDACT_VALUE,
        "on_metadata_drop": raw.get("on_metadata_drop"),
    }


def _metadata_rule_matches(rules: set[str], key_path: str, key: str) -> bool:
    normalized_path = str(key_path or "").strip().lower()
    normalized_key = str(key or "").strip().lower()
    return normalized_path in rules or normalized_key in rules


def _is_sensitive_metadata_key(key_path: str, key: str) -> bool:
    haystacks = [str(key_path or "").lower(), str(key or "").lower()]
    return any(pattern in haystack for pattern in DEFAULT_SENSITIVE_METADATA_KEY_PATTERNS for haystack in haystacks)


def _hash_metadata_value(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return f"sha256_{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _emit_metadata_drop(policy: Dict[str, Any], key_path: str, reason: str) -> None:
    callback = policy.get("on_metadata_drop")
    if callable(callback):
        callback({
            "key_path": key_path,
            "reason": reason,
            "mode": policy["mode"],
        })


def _sanitize_metadata_value(value: Any, key_path: str, key: str, policy: Dict[str, Any]) -> Any:
    if value is None:
        return None
    if _metadata_rule_matches(policy["denylist_keys"], key_path, key):
        _emit_metadata_drop(policy, key_path, "denylist")
        return None
    if policy["mode"] == "allowlisted_metadata" and not _metadata_rule_matches(policy["allowlist_keys"], key_path, key):
        _emit_metadata_drop(policy, key_path, "allowlist")
        return None
    if policy["mode"] == "strict_finops" and key.strip().lower() not in STRICT_FINOPS_METADATA_KEYS:
        _emit_metadata_drop(policy, key_path, "allowlist")
        return None
    if _metadata_rule_matches(policy["hash_keys"], key_path, key):
        _emit_metadata_drop(policy, key_path, "hashed")
        return _hash_metadata_value(value)
    if _metadata_rule_matches(policy["redact_keys"], key_path, key) or _is_sensitive_metadata_key(key_path, key):
        _emit_metadata_drop(policy, key_path, "redacted")
        return policy["redact_value"]
    if isinstance(value, str):
        if len(value) > policy["max_value_length"]:
            _emit_metadata_drop(policy, key_path, "truncated")
            return f"{value[:policy['max_value_length']]}…"
        return value
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, datetime):
        return _to_iso(value)
    if isinstance(value, list):
        sanitized = [
            _sanitize_metadata_value(item, f"{key_path}[{index}]", key, policy)
            for index, item in enumerate(value)
        ]
        return [item for item in sanitized if item is not None] or None
    if isinstance(value, dict):
        sanitized = _sanitize_custom_metadata(value, policy, key_path)
        return sanitized or None
    _emit_metadata_drop(policy, key_path, "unsupported_value")
    return None


def _sanitize_custom_metadata(
    metadata: Optional[Dict[str, Any]],
    policy: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    result: Dict[str, Any] = {}
    accepted_keys = 0
    for raw_key, raw_value in metadata.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        key_path = f"{prefix}.{key}" if prefix else key
        if accepted_keys >= policy["max_keys"]:
            _emit_metadata_drop(policy, key_path, "max_keys")
            continue
        sanitized = _sanitize_metadata_value(raw_value, key_path, key, policy)
        if sanitized is None:
            continue
        candidate = {**result, key: sanitized}
        if len(json.dumps(candidate, default=str)) > policy["max_serialized_bytes"]:
            _emit_metadata_drop(policy, key_path, "max_serialized_bytes")
            continue
        result[key] = sanitized
        accepted_keys += 1
    return result


def _should_attach_default_otlp_authorization(otlp_url: str) -> bool:
    try:
        hostname = (urlparse(otlp_url).hostname or "").lower()
    except Exception:
        return False
    return hostname == "cloptima.ai" or hostname.endswith(".cloptima.ai")


def _unix_nano_string(value: Optional[Union[datetime, str]], fallback_ms: int) -> str:
    iso = _to_iso(value)
    if iso:
        parsed = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return str(int(parsed.timestamp() * 1_000_000_000))
    return str(int(fallback_ms) * 1_000_000)


def _otlp_attribute_value(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": value}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (dict, list)):
        return {"stringValue": json.dumps(value, default=str)}
    return None


def _otlp_attributes_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    pairs: List[tuple[str, Any]] = [
        ("gen_ai.system", _clean_str(payload.get("provider"))),
        ("gen_ai.request.model", _clean_str(payload.get("model"))),
        ("gen_ai.response.model", _clean_str(payload.get("model"))),
        ("gen_ai.request.id", _clean_str(payload.get("request_id"))),
        ("gen_ai.response.id", _clean_str(payload.get("provider_request_id"))),
        ("source_event_id", _clean_str(payload.get("source_event_id"))),
        ("gen_ai.usage.input_tokens", _clean_int(payload.get("input_tokens"))),
        ("gen_ai.usage.output_tokens", _clean_int(payload.get("output_tokens"))),
        ("gen_ai.usage.total_tokens", _clean_int(payload.get("total_tokens"))),
        ("gen_ai.usage.reasoning_tokens", _clean_int(payload.get("reasoning_tokens"))),
        ("gen_ai.usage.cached_input_tokens", _clean_int(payload.get("cached_input_tokens"))),
        ("gen_ai.usage.cost", _clean_float(payload.get("vendor_reported_cost_usd"))),
        ("cache_hit", payload.get("cache_hit") if isinstance(payload.get("cache_hit"), bool) else None),
        ("cloptima.request_id", _clean_str(payload.get("request_id"))),
        ("trace_id", _clean_str(payload.get("trace_id"))),
    ]
    for key, value in metadata.items():
        pairs.append((str(key), value))
    attributes: List[Dict[str, Any]] = []
    seen = set()
    for key, value in pairs:
        if not key or key in seen:
            continue
        seen.add(key)
        attr_value = _otlp_attribute_value(value)
        if attr_value is not None:
            attributes.append({"key": key, "value": attr_value})
    return attributes


def _payload_to_otlp_request(
    payload: Dict[str, Any],
    sdk_name: str,
    sdk_version: Optional[str],
    service_name: str,
    service_version: Optional[str],
) -> Dict[str, Any]:
    events = payload.get("events") if isinstance(payload.get("events"), list) else [payload]
    now_ms = int(time.time() * 1000)
    spans: List[Dict[str, Any]] = []
    for event in events:
        event_payload = event if isinstance(event, dict) else {}
        latency_ms = _clean_int(event_payload.get("latency_ms")) or 0
        start_nanos = _unix_nano_string(event_payload.get("started_at"), now_ms)
        end_nanos = _unix_nano_string(event_payload.get("completed_at"), now_ms + latency_ms)
        spans.append(
            {
                "traceId": uuid.uuid4().hex,
                "spanId": uuid.uuid4().hex[:16],
                "name": f"llm.{_clean_str(event_payload.get('provider')) or 'unknown'}.{_clean_str(event_payload.get('model')) or 'unknown'}",
                "kind": 3,
                "startTimeUnixNano": start_nanos,
                "endTimeUnixNano": end_nanos,
                "attributes": _otlp_attributes_from_payload(event_payload),
                "status": _strip_none(
                    {
                        "code": 2 if _clean_str(event_payload.get("status")) == "failed" else 1,
                        "message": _clean_str(event_payload.get("error_message")),
                    }
                ),
            }
        )
    resource_attributes = [{"key": "service.name", "value": {"stringValue": service_name}}]
    if service_version:
        resource_attributes.append({"key": "service.version", "value": {"stringValue": service_version}})
    scope: Dict[str, Any] = {"name": sdk_name}
    if sdk_version:
        scope["version"] = sdk_version
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": resource_attributes},
                "scopeSpans": [
                    {
                        "scope": scope,
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def _preview_default_attribution(
    default_attribution: Optional[Union[LLMAttribution, Dict[str, Any]]],
) -> LLMAttribution:
    if isinstance(default_attribution, LLMAttribution):
        return default_attribution
    raw = default_attribution or {}
    return LLMAttribution(
        app_id=_clean_str(raw.get("app_id")) or "preview-app",
        environment=_clean_str(raw.get("environment")) or "preview",
        team_id=_clean_str(raw.get("team_id")),
        feature_id=_clean_str(raw.get("feature_id")),
        workflow_id=_clean_str(raw.get("workflow_id")),
        business_unit=_clean_str(raw.get("business_unit")),
        cost_center=_clean_str(raw.get("cost_center")),
        product=_clean_str(raw.get("product")),
        customer_segment=_clean_str(raw.get("customer_segment")),
        end_customer_id=_clean_str(raw.get("end_customer_id")),
        tenant_id=_clean_str(raw.get("tenant_id")),
        release=_clean_str(raw.get("release")),
        actor_id=_clean_str(raw.get("actor_id")),
        actor_type=_clean_str(raw.get("actor_type")),
    )


def _validate_single_payload(payload: Dict[str, Any], prefix: str) -> List[str]:
    errors: List[str] = []
    status = _clean_str(payload.get("status"))
    metadata = payload.get("metadata")
    if _clean_str(payload.get("schema_version")) != SDK_EVENT_SCHEMA_VERSION:
        errors.append(f"{prefix}.schema_version must equal {SDK_EVENT_SCHEMA_VERSION}")
    if not _clean_str(payload.get("sdk_name")):
        errors.append(f"{prefix}.sdk_name is required")
    if not _clean_str(payload.get("source_event_id")):
        errors.append(f"{prefix}.source_event_id is required")
    if not _clean_str(payload.get("provider")):
        errors.append(f"{prefix}.provider is required")
    if not _clean_str(payload.get("model")):
        errors.append(f"{prefix}.model is required")
    if status and status not in {"succeeded", "failed", "partial", "blocked"}:
        errors.append(f"{prefix}.status must be one of succeeded, failed, partial, blocked")
    for key in ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens", "cached_input_tokens", "latency_ms"):
        value = payload.get(key)
        if value is not None and _clean_int(value) is None:
            errors.append(f"{prefix}.{key} must be a finite non-negative integer when set")
    if metadata is not None and not isinstance(metadata, dict):
        errors.append(f"{prefix}.metadata must be an object when set")
    return errors


def preview_event_payload(
    event: Union[LLMUsageEvent, Mapping[str, Any]],
    *,
    default_attribution: Optional[Union[LLMAttribution, Dict[str, Any]]] = None,
    sdk_name: str = "cloptima-llm-observability",
    sdk_version: Optional[str] = None,
    metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return _event_payload(
        _coerce_usage_event_input(event),
        _preview_default_attribution(default_attribution),
        sdk_name,
        _resolve_metadata_privacy_policy(metadata_policy),
        sdk_version,
    )


def preview_batch_payload(
    events: List[Union[LLMUsageEvent, Mapping[str, Any]]],
    *,
    default_attribution: Optional[Union[LLMAttribution, Dict[str, Any]]] = None,
    sdk_name: str = "cloptima-llm-observability",
    sdk_version: Optional[str] = None,
    metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    payloads = [
        preview_event_payload(
            event,
            default_attribution=default_attribution,
            sdk_name=sdk_name,
            sdk_version=sdk_version,
            metadata_policy=metadata_policy,
        )
        for event in events
    ]
    if len(payloads) == 1:
        return payloads[0]
    return {"schema_version": SDK_BATCH_SCHEMA_VERSION, "events": payloads}


def try_extract_usage(input_value: Any, *extractors: Callable[[Any], Dict[str, Any]]) -> Dict[str, Any]:
    for extractor in extractors:
        if extractor is None:
            continue
        try:
            extracted = extractor(input_value) or {}
        except Exception:
            continue
        if _has_meaningful_extraction(extracted):
            return _strip_none(dict(extracted))
    return {}


def compose_usage_extractors(*extractors: Callable[[Any], Dict[str, Any]]) -> Callable[[Any], Dict[str, Any]]:
    def _composed(input_value: Any) -> Dict[str, Any]:
        return try_extract_usage(input_value, *extractors)

    return _composed


def with_usage_overrides(
    extractor: Callable[[Any], Dict[str, Any]],
    overrides: Union[Dict[str, Any], Callable[[Dict[str, Any], Any], Dict[str, Any]]],
) -> Callable[[Any], Dict[str, Any]]:
    def _wrapped(input_value: Any) -> Dict[str, Any]:
        extracted = extractor(input_value) or {}
        override_values = overrides(extracted, input_value) if callable(overrides) else overrides
        return _strip_none({**extracted, **(override_values or {})})

    return _wrapped


def create_mapped_usage_extractor(
    *,
    defaults: Optional[Dict[str, Any]] = None,
    fields: Optional[Dict[str, Union[str, List[str], Tuple[str, ...]]]] = None,
    number_fields: Optional[Dict[str, Union[str, List[str], Tuple[str, ...]]]] = None,
    boolean_fields: Optional[Dict[str, Union[str, List[str], Tuple[str, ...]]]] = None,
    extra_usage_units: Optional[Dict[str, Union[str, List[str], Tuple[str, ...]]]] = None,
    metadata: Optional[Dict[str, Union[str, List[str], Tuple[str, ...]]]] = None,
) -> UsageExtractor:
    def _extract(input_value: Any) -> Dict[str, Any]:
        record = _coerce_mapping(input_value)
        extracted: Dict[str, Any] = dict(defaults or {})
        for key, path in (fields or {}).items():
            value = _resolve_mapped_value(record, path)
            if key == "vendor_reported_cost_usd":
                extracted[key] = _clean_float(value) if _clean_float(value) is not None else _clean_str(value)
            else:
                extracted[key] = _clean_str(value)
        for key, path in (number_fields or {}).items():
            extracted[key] = _clean_int(_resolve_mapped_value(record, path))
        for key, path in (boolean_fields or {}).items():
            if key == "cache_hit":
                extracted[key] = _clean_bool(_resolve_mapped_value(record, path))
        if extra_usage_units:
            extracted["extra_usage_units"] = _clean_usage_map(
                {key: _resolve_mapped_value(record, path) for key, path in extra_usage_units.items()}
            ) or None
        if metadata:
            extracted["metadata"] = _strip_none(
                {key: _resolve_mapped_value(record, path) for key, path in metadata.items()}
            ) or None
        if extracted.get("total_tokens") is None and (
            extracted.get("input_tokens") is not None or extracted.get("output_tokens") is not None
        ):
            extracted["total_tokens"] = (extracted.get("input_tokens") or 0) + (extracted.get("output_tokens") or 0)
        return _strip_none(extracted)

    return _extract


def _merge_optional_dicts(*values: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    merged: Dict[str, Any] = {}
    for value in values:
        if value:
            merged.update(value)
    return merged or None


def create_observed_call(
    client: Union["CloptimaLLMObservability", "DisabledCloptimaLLMObservability"],
    *,
    provider: str,
    model: str,
    extract_usage: Optional[Callable[[Any], Dict[str, Any]]] = None,
    attribution: Optional[Dict[str, Any]] = None,
    agent: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metadata_policy: Optional[Union["MetadataPrivacyPolicy", Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    fire_and_forget: bool = True,
) -> Callable[..., Any]:
    def _invoke(call: Callable[[], Any], **overrides: Any) -> Any:
        return client.observe(
            provider=overrides.get("provider") or provider,
            model=overrides.get("model") or model,
            call=call,
            extract_usage=overrides.get("extract_usage") or extract_usage,
            attribution=_merge_optional_dicts(attribution, overrides.get("attribution")),
            agent=_merge_optional_dicts(agent, overrides.get("agent")),
            metadata=_merge_optional_dicts(metadata, overrides.get("metadata")),
            metadata_policy=overrides.get("metadata_policy", metadata_policy),
            request_id=overrides.get("request_id") or request_id,
            trace_id=overrides.get("trace_id") or trace_id,
            fire_and_forget=overrides.get("fire_and_forget", fire_and_forget),
        )

    return _invoke


def create_observed_async_call(
    client: Union["CloptimaLLMObservability", "DisabledCloptimaLLMObservability"],
    *,
    provider: str,
    model: str,
    extract_usage: Optional[Callable[[Any], Dict[str, Any]]] = None,
    attribution: Optional[Dict[str, Any]] = None,
    agent: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metadata_policy: Optional[Union["MetadataPrivacyPolicy", Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    fire_and_forget: bool = True,
) -> Callable[..., Any]:
    async def _invoke(call: Callable[[], Any], **overrides: Any) -> Any:
        return await client.observe_async(
            provider=overrides.get("provider") or provider,
            model=overrides.get("model") or model,
            call=call,
            extract_usage=overrides.get("extract_usage") or extract_usage,
            attribution=_merge_optional_dicts(attribution, overrides.get("attribution")),
            agent=_merge_optional_dicts(agent, overrides.get("agent")),
            metadata=_merge_optional_dicts(metadata, overrides.get("metadata")),
            metadata_policy=overrides.get("metadata_policy", metadata_policy),
            request_id=overrides.get("request_id") or request_id,
            trace_id=overrides.get("trace_id") or trace_id,
            fire_and_forget=overrides.get("fire_and_forget", fire_and_forget),
        )

    return _invoke


def create_observed_stream(
    client: Union["CloptimaLLMObservability", "DisabledCloptimaLLMObservability"],
    *,
    provider: str,
    model: str,
    extract_usage: Optional[Callable[[List[Any]], Dict[str, Any]]] = None,
    attribution: Optional[Dict[str, Any]] = None,
    agent: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metadata_policy: Optional[Union["MetadataPrivacyPolicy", Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    fire_and_forget: bool = False,
    max_buffered_chunks: int = 256,
) -> Callable[..., Iterator[Any]]:
    def _invoke(call: Callable[[], Iterable[Any]], **overrides: Any) -> Iterator[Any]:
        return client.observe_stream(
            provider=overrides.get("provider") or provider,
            model=overrides.get("model") or model,
            call=call,
            extract_usage=overrides.get("extract_usage") or extract_usage,
            attribution=_merge_optional_dicts(attribution, overrides.get("attribution")),
            agent=_merge_optional_dicts(agent, overrides.get("agent")),
            metadata=_merge_optional_dicts(metadata, overrides.get("metadata")),
            metadata_policy=overrides.get("metadata_policy", metadata_policy),
            request_id=overrides.get("request_id") or request_id,
            trace_id=overrides.get("trace_id") or trace_id,
            fire_and_forget=overrides.get("fire_and_forget", fire_and_forget),
            max_buffered_chunks=overrides.get("max_buffered_chunks", max_buffered_chunks),
        )

    return _invoke


def create_observed_async_stream(
    client: Union["CloptimaLLMObservability", "DisabledCloptimaLLMObservability"],
    *,
    provider: str,
    model: str,
    extract_usage: Optional[Callable[[List[Any]], Dict[str, Any]]] = None,
    attribution: Optional[Dict[str, Any]] = None,
    agent: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metadata_policy: Optional[Union["MetadataPrivacyPolicy", Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    fire_and_forget: bool = False,
    max_buffered_chunks: int = 256,
) -> Callable[..., AsyncIterator[Any]]:
    def _invoke(call: Callable[[], Any], **overrides: Any) -> AsyncIterator[Any]:
        return client.observe_async_stream(
            provider=overrides.get("provider") or provider,
            model=overrides.get("model") or model,
            call=call,
            extract_usage=overrides.get("extract_usage") or extract_usage,
            attribution=_merge_optional_dicts(attribution, overrides.get("attribution")),
            agent=_merge_optional_dicts(agent, overrides.get("agent")),
            metadata=_merge_optional_dicts(metadata, overrides.get("metadata")),
            metadata_policy=overrides.get("metadata_policy", metadata_policy),
            request_id=overrides.get("request_id") or request_id,
            trace_id=overrides.get("trace_id") or trace_id,
            fire_and_forget=overrides.get("fire_and_forget", fire_and_forget),
            max_buffered_chunks=overrides.get("max_buffered_chunks", max_buffered_chunks),
        )

    return _invoke


def bind_observed_call(
    client: Union["CloptimaLLMObservability", "DisabledCloptimaLLMObservability"],
    method: Callable[..., Any],
    *,
    provider: str,
    model: str,
    extract_usage: Optional[Callable[[Any], Dict[str, Any]]] = None,
    attribution: Optional[Dict[str, Any]] = None,
    agent: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metadata_policy: Optional[Union["MetadataPrivacyPolicy", Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    fire_and_forget: bool = True,
    resolve_overrides: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Callable[..., Any]:
    observed = create_observed_call(
        client,
        provider=provider,
        model=model,
        extract_usage=extract_usage,
        attribution=attribution,
        agent=agent,
        metadata=metadata,
        metadata_policy=metadata_policy,
        request_id=request_id,
        trace_id=trace_id,
        fire_and_forget=fire_and_forget,
    )

    @functools.wraps(method)
    def _invoke(*args: Any, **kwargs: Any) -> Any:
        overrides = resolve_overrides(*args, **kwargs) if resolve_overrides else {}
        return observed(lambda: method(*args, **kwargs), **overrides)

    return _invoke


def bind_observed_async_call(
    client: Union["CloptimaLLMObservability", "DisabledCloptimaLLMObservability"],
    method: Callable[..., Any],
    *,
    provider: str,
    model: str,
    extract_usage: Optional[Callable[[Any], Dict[str, Any]]] = None,
    attribution: Optional[Dict[str, Any]] = None,
    agent: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metadata_policy: Optional[Union["MetadataPrivacyPolicy", Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    fire_and_forget: bool = True,
    resolve_overrides: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Callable[..., Any]:
    observed = create_observed_async_call(
        client,
        provider=provider,
        model=model,
        extract_usage=extract_usage,
        attribution=attribution,
        agent=agent,
        metadata=metadata,
        metadata_policy=metadata_policy,
        request_id=request_id,
        trace_id=trace_id,
        fire_and_forget=fire_and_forget,
    )

    @functools.wraps(method)
    async def _invoke(*args: Any, **kwargs: Any) -> Any:
        overrides = resolve_overrides(*args, **kwargs) if resolve_overrides else {}
        return await observed(lambda: method(*args, **kwargs), **overrides)

    return _invoke


def bind_observed_stream(
    client: Union["CloptimaLLMObservability", "DisabledCloptimaLLMObservability"],
    method: Callable[..., Iterable[Any]],
    *,
    provider: str,
    model: str,
    extract_usage: Optional[Callable[[List[Any]], Dict[str, Any]]] = None,
    attribution: Optional[Dict[str, Any]] = None,
    agent: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metadata_policy: Optional[Union["MetadataPrivacyPolicy", Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    fire_and_forget: bool = False,
    max_buffered_chunks: int = 256,
    resolve_overrides: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Callable[..., Iterator[Any]]:
    observed = create_observed_stream(
        client,
        provider=provider,
        model=model,
        extract_usage=extract_usage,
        attribution=attribution,
        agent=agent,
        metadata=metadata,
        metadata_policy=metadata_policy,
        request_id=request_id,
        trace_id=trace_id,
        fire_and_forget=fire_and_forget,
        max_buffered_chunks=max_buffered_chunks,
    )

    @functools.wraps(method)
    def _invoke(*args: Any, **kwargs: Any) -> Iterator[Any]:
        overrides = resolve_overrides(*args, **kwargs) if resolve_overrides else {}
        return observed(lambda: method(*args, **kwargs), **overrides)

    return _invoke


def bind_observed_async_stream(
    client: Union["CloptimaLLMObservability", "DisabledCloptimaLLMObservability"],
    method: Callable[..., Any],
    *,
    provider: str,
    model: str,
    extract_usage: Optional[Callable[[List[Any]], Dict[str, Any]]] = None,
    attribution: Optional[Dict[str, Any]] = None,
    agent: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metadata_policy: Optional[Union["MetadataPrivacyPolicy", Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    fire_and_forget: bool = False,
    max_buffered_chunks: int = 256,
    resolve_overrides: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Callable[..., AsyncIterator[Any]]:
    observed = create_observed_async_stream(
        client,
        provider=provider,
        model=model,
        extract_usage=extract_usage,
        attribution=attribution,
        agent=agent,
        metadata=metadata,
        metadata_policy=metadata_policy,
        request_id=request_id,
        trace_id=trace_id,
        fire_and_forget=fire_and_forget,
        max_buffered_chunks=max_buffered_chunks,
    )

    @functools.wraps(method)
    def _invoke(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        overrides = resolve_overrides(*args, **kwargs) if resolve_overrides else {}
        return observed(lambda: method(*args, **kwargs), **overrides)

    return _invoke


def wrap_observed_service(
    client: Union["CloptimaLLMObservability", "DisabledCloptimaLLMObservability"],
    service: Any,
    bindings: Dict[str, Dict[str, Any]],
) -> Any:
    class _WrappedService:
        def __init__(self, original_service: Any) -> None:
            object.__setattr__(self, "_original_service", original_service)

        def __getattr__(self, name: str) -> Any:
            return getattr(object.__getattribute__(self, "_original_service"), name)

        def __setattr__(self, name: str, value: Any) -> None:
            object.__setattr__(self, name, value)

        def __dir__(self) -> List[str]:
            original = object.__getattribute__(self, "_original_service")
            return sorted(set(object.__dir__(self) + dir(original)))

    wrapped = _WrappedService(service)
    for method_name, binding in bindings.items():
        original = getattr(service, method_name, None)
        if not callable(original):
            raise ValueError(f"Cannot wrap non-callable service method: {method_name}")
        kind = binding.get("kind")
        options = dict(binding.get("options") or {})
        resolve_overrides = binding.get("resolve_overrides")
        if kind == "call":
            wrapped_method = bind_observed_call(client, original, resolve_overrides=resolve_overrides, **options)
        elif kind == "async_call":
            wrapped_method = bind_observed_async_call(client, original, resolve_overrides=resolve_overrides, **options)
        elif kind == "stream":
            wrapped_method = bind_observed_stream(client, original, resolve_overrides=resolve_overrides, **options)
        elif kind == "async_stream":
            wrapped_method = bind_observed_async_stream(client, original, resolve_overrides=resolve_overrides, **options)
        else:
            raise ValueError(f"Unsupported observed service binding kind: {kind}")
        setattr(wrapped, method_name, wrapped_method)
    return wrapped


def preview_otlp_request(
    payload: Dict[str, Any],
    *,
    sdk_name: str = "cloptima-llm-observability",
    sdk_version: Optional[str] = None,
    service_name: str = "cloptima-llm-observability",
    service_version: Optional[str] = None,
) -> Dict[str, Any]:
    return _payload_to_otlp_request(
        payload,
        sdk_name,
        sdk_version,
        service_name,
        service_version,
    )


def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    events = payload.get("events") if isinstance(payload.get("events"), list) else [payload]
    errors: List[str] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"events[{index}] must be an object")
            continue
        errors.extend(_validate_single_payload(event, f"events[{index}]"))
    if isinstance(payload.get("events"), list) and _clean_str(payload.get("schema_version")) != SDK_BATCH_SCHEMA_VERSION:
        errors.insert(0, f"batch.schema_version must equal {SDK_BATCH_SCHEMA_VERSION}")
    return {"valid": len(errors) == 0, "errors": errors}


class CloptimaLLMObservability:
    def __init__(
        self,
        *,
        api_base_url: Optional[str] = None,
        api_key: str,
        default_attribution: LLMAttribution,
        delivery_mode: str = "cloptima_http",
        otlp_headers: Optional[Dict[str, str]] = None,
        otlp_service_name: str = "cloptima-llm-observability",
        otlp_service_version: Optional[str] = None,
        sdk_name: str = "cloptima-llm-observability",
        sdk_version: Optional[str] = None,
        timeout_seconds: float = 3.0,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
        on_drop: Optional[Callable[[LLMUsageEvent, str], None]] = None,
        async_queue_max_size: int = 1000,
        async_batch_size: int = 20,
        async_flush_interval_seconds: float = 0.25,
        async_retry_count: int = 2,
        async_retry_backoff_seconds: float = 0.1,
        async_retry_jitter_ratio: float = 0.2,
        async_http_client: Optional[Any] = None,
    ) -> None:
        self.api_base_url = _resolve_api_base_url(api_base_url)
        self.ingest_url = _resolve_ingest_url(self.api_base_url)
        self.api_key = api_key
        self.default_attribution = default_attribution
        self.delivery_mode = _resolve_delivery_mode(delivery_mode)
        self.otlp_url = _resolve_otlp_url(self.api_base_url)
        self.otlp_headers = otlp_headers or {}
        self.otlp_service_name = otlp_service_name
        self.otlp_service_version = otlp_service_version
        self.sdk_name = sdk_name
        self.sdk_version = sdk_version
        self.timeout_seconds = timeout_seconds
        self.metadata_privacy_policy = _resolve_metadata_privacy_policy(metadata_policy)
        self.on_error = on_error
        self.on_drop = on_drop
        self.async_batch_size = max(1, int(async_batch_size or 1))
        self.async_flush_interval_seconds = max(0.0, float(async_flush_interval_seconds or 0.0))
        self.async_retry_count = max(0, int(async_retry_count or 0))
        self.async_retry_backoff_seconds = max(0.0, float(async_retry_backoff_seconds or 0.0))
        self.async_retry_jitter_ratio = min(1.0, max(0.0, float(async_retry_jitter_ratio or 0.0)))
        self._async_queue: queue.Queue[Optional[_QueuedEvent]] = queue.Queue(maxsize=max(1, int(async_queue_max_size or 1)))
        self._worker_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._closed = False
        self._stats_lock = threading.Lock()
        self._dropped_events = 0
        self._delivered_events = 0
        self._failed_batches = 0
        self._async_http_client = async_http_client
        self._owns_async_http_client = async_http_client is None
        self._atexit_registered = False

    def is_enabled(self) -> bool:
        return True

    def get_init_error(self) -> Optional[BaseException]:
        return None

    def run_with_attribution(
        self,
        call: Callable[[], T],
        attribution: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        return run_with_attribution(call, attribution, **kwargs)

    def with_attribution(
        self,
        attribution: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        return with_attribution(attribution, **kwargs)

    def run_with_workflow(
        self,
        call: Callable[[], T],
        name: Optional[str] = None,
        attribution: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        return run_with_workflow(call, name=name, attribution=attribution, **kwargs)

    def with_workflow(
        self,
        name: Optional[str] = None,
        attribution: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        return with_workflow(name=name, attribution=attribution, **kwargs)

    def run_with_task(
        self,
        call: Callable[[], T],
        name: Optional[str] = None,
        attribution: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        return run_with_task(call, name=name, attribution=attribution, **kwargs)

    def with_task(
        self,
        name: Optional[str] = None,
        attribution: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        return with_task(name=name, attribution=attribution, **kwargs)

    def _otlp_request_headers(self) -> Dict[str, str]:
        headers = {
            "content-type": "application/json",
            **self.otlp_headers,
        }
        if not any(key.lower() == "authorization" for key in headers) and _should_attach_default_otlp_authorization(self.otlp_url):
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def record(self, event: LLMUsageEvent) -> None:
        self._post_payload(self._event_payload(event))

    def record_batch(self, events: List[LLMUsageEvent]) -> None:
        payload = self._batch_payload(events)
        if payload is None:
            return
        self._post_payload(payload)

    def _batch_payload(self, events: List[LLMUsageEvent]) -> Optional[Dict[str, Any]]:
        payloads = [
            self._event_payload(event)
            for event in events
        ]
        if not payloads:
            return None
        if len(payloads) == 1:
            return payloads[0]
        return {"schema_version": SDK_BATCH_SCHEMA_VERSION, "events": payloads}

    def _payload_with_envelope_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        decorated = dict(payload)
        decorated["sdk_delivery_stats"] = _delivery_stats_payload(self.stats())
        if isinstance(payload.get("events"), list):
            decorated["batch_schema_version"] = SDK_BATCH_SCHEMA_VERSION
        return decorated

    def _event_payload(
        self,
        event: LLMUsageEvent,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return _event_payload(
            event,
            self.default_attribution,
            self.sdk_name,
            _resolve_metadata_privacy_policy(metadata_policy) if metadata_policy is not None else self.metadata_privacy_policy,
            self.sdk_version,
        )

    def _post_payload(self, payload: Dict[str, Any]) -> None:
        payload = self._payload_with_envelope_metadata(payload)
        def _with_retries(call: Callable[[], None]) -> None:
            attempts = self.async_retry_count + 1
            for attempt in range(attempts):
                try:
                    call()
                    return
                except BaseException:
                    if attempt + 1 >= attempts:
                        raise
                    time.sleep(self._retry_delay_seconds(attempt))

        cloptima_error: Optional[BaseException] = None
        if self.delivery_mode in {"cloptima_http", INTERNAL_DUAL_DELIVERY_MODE}:
            try:
                _with_retries(
                    lambda: self._post_json_once(
                        self.ingest_url,
                        payload,
                        {
                            "authorization": f"Bearer {self.api_key}",
                            "content-type": "application/json",
                        },
                        "Cloptima LLM ingest",
                    )
                )
            except BaseException as exc:
                cloptima_error = exc
                if self.delivery_mode == INTERNAL_DUAL_DELIVERY_MODE and self.on_error:
                    self.on_error(exc)
        if self.delivery_mode in {"otlp_http", INTERNAL_DUAL_DELIVERY_MODE}:
            otlp_payload = _payload_to_otlp_request(payload, self.sdk_name, self.sdk_version, self.otlp_service_name, self.otlp_service_version)
            try:
                _with_retries(
                    lambda: self._post_json_once(
                        self.otlp_url,
                        otlp_payload,
                        self._otlp_request_headers(),
                        "Cloptima OTLP ingest",
                    )
                )
            except BaseException as exc:
                if self.delivery_mode == "otlp_http":
                    raise
                if self.on_error:
                    self.on_error(exc)
                if cloptima_error is not None:
                    raise cloptima_error
                return
        if cloptima_error is not None:
            raise cloptima_error

    def _retry_delay_seconds(self, attempt: int) -> float:
        base_delay = self.async_retry_backoff_seconds * (2 ** attempt)
        if base_delay <= 0 or self.async_retry_jitter_ratio <= 0:
            return base_delay
        return base_delay + random.uniform(0, base_delay * self.async_retry_jitter_ratio)

    def _post_json_once(self, url: str, payload: Dict[str, Any], headers: Dict[str, str], label: str) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                if response.status >= 400:
                    raise RuntimeError(f"{label} failed with HTTP {response.status}")
                response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"{label} failed with HTTP {exc.code}") from exc

    def _event_body(self, event: LLMUsageEvent) -> bytes:
        return json.dumps(
            _event_payload(event, self.default_attribution, self.sdk_name, self.metadata_privacy_policy, self.sdk_version),
            default=str,
        ).encode("utf-8")

    def _payload_body(self, payload: Dict[str, Any]) -> bytes:
        return json.dumps(payload, default=str).encode("utf-8")

    async def _get_async_http_client(self) -> Any:
        if self._async_http_client is not None:
            if not getattr(self._async_http_client, "is_closed", False):
                return self._async_http_client
            if not self._owns_async_http_client:
                raise RuntimeError(
                    "The provided async_http_client is closed; provide a new open client or omit async_http_client"
                )

        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for async Cloptima LLM observability; install the optional 'httpx' extra"
            ) from exc

        self._async_http_client = httpx.AsyncClient(timeout=self.timeout_seconds)
        self._owns_async_http_client = True
        return self._async_http_client

    async def arecord(self, event: LLMUsageEvent) -> None:
        await self._apost_payload(_event_payload(event, self.default_attribution, self.sdk_name, self.metadata_privacy_policy, self.sdk_version))

    async def arecord_batch(self, events: List[LLMUsageEvent]) -> None:
        payload = self._batch_payload(events)
        if payload is None:
            return
        await self._apost_payload(payload)

    async def _apost_payload(self, payload: Dict[str, Any]) -> None:
        payload = self._payload_with_envelope_metadata(payload)
        client = await self._get_async_http_client()
        cloptima_error: Optional[BaseException] = None
        if self.delivery_mode in {"cloptima_http", INTERNAL_DUAL_DELIVERY_MODE}:
            try:
                response = await client.post(
                    self.ingest_url,
                    content=self._payload_body(payload),
                    headers={
                        "authorization": f"Bearer {self.api_key}",
                        "content-type": "application/json",
                    },
                )
                if response.status_code >= 400:
                    raise RuntimeError(f"Cloptima LLM ingest failed with HTTP {response.status_code}")
                await response.aread()
            except BaseException as exc:
                cloptima_error = exc
                if self.delivery_mode == INTERNAL_DUAL_DELIVERY_MODE and self.on_error:
                    self.on_error(exc)
        if self.delivery_mode in {"otlp_http", INTERNAL_DUAL_DELIVERY_MODE}:
            otlp_payload = _payload_to_otlp_request(payload, self.sdk_name, self.sdk_version, self.otlp_service_name, self.otlp_service_version)
            try:
                response = await client.post(
                    self.otlp_url,
                    content=self._payload_body(otlp_payload),
                    headers=self._otlp_request_headers(),
                )
                if response.status_code >= 400:
                    raise RuntimeError(f"Cloptima OTLP ingest failed with HTTP {response.status_code}")
                await response.aread()
            except BaseException as exc:
                if self.delivery_mode == "otlp_http":
                    raise
                if self.on_error:
                    self.on_error(exc)
                if cloptima_error is not None:
                    raise cloptima_error
                return
        if cloptima_error is not None:
            raise cloptima_error

    def record_async(self, event: LLMUsageEvent) -> None:
        self._record_async_with_policy(event)

    def _record_async_with_policy(
        self,
        event: LLMUsageEvent,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
    ) -> None:
        if self._closed:
            self._record_drop(event, "client_closed")
            if self.on_error:
                self.on_error(RuntimeError("Cloptima LLM observability client is closed"))
            return
        self._ensure_worker()
        try:
            self._async_queue.put_nowait(_QueuedEvent(event=event, metadata_policy=metadata_policy))
        except queue.Full as exc:
            self._record_drop(event, "queue_full")
            if self.on_error:
                self.on_error(RuntimeError("Cloptima LLM observability async queue is full"))

    def stats(self) -> CloptimaLLMClientStats:
        with self._stats_lock:
            return CloptimaLLMClientStats(
                queued_events=self._async_queue.qsize(),
                dropped_events=self._dropped_events,
                delivered_events=self._delivered_events,
                failed_batches=self._failed_batches,
            )

    def _record_drop(self, event: LLMUsageEvent, reason: str) -> None:
        with self._stats_lock:
            self._dropped_events += 1
        if self.on_drop:
            self.on_drop(event, reason)

    def _record_delivered(self, count: int) -> None:
        with self._stats_lock:
            self._delivered_events += count

    def _record_failed_batch(self) -> None:
        with self._stats_lock:
            self._failed_batches += 1

    def flush(self, timeout_seconds: Optional[float] = None) -> bool:
        deadline = None if timeout_seconds is None else time.monotonic() + max(0.0, timeout_seconds)
        with self._async_queue.all_tasks_done:
            while self._async_queue.unfinished_tasks:
                if deadline is None:
                    self._async_queue.all_tasks_done.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._async_queue.all_tasks_done.wait(timeout=remaining)
            return True

    def close(self, timeout_seconds: Optional[float] = 5.0) -> bool:
        self._closed = True
        worker = self._worker
        if worker is None:
            return True
        try:
            self._async_queue.put_nowait(None)
        except queue.Full:
            if not self.flush(timeout_seconds):
                return False
            self._async_queue.put(None)
        worker.join(timeout_seconds)
        return not worker.is_alive()

    async def aclose(self, timeout_seconds: Optional[float] = 5.0) -> bool:
        closed = self.close(timeout_seconds)
        client = self._async_http_client
        if client is not None and self._owns_async_http_client and hasattr(client, "aclose"):
            await client.aclose()
            self._async_http_client = None
        return closed

    def _ensure_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        with self._worker_lock:
            if self._worker and self._worker.is_alive():
                return
            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True
            self._worker = threading.Thread(target=self._worker_loop, name="cloptima-llm-observability", daemon=True)
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            item = self._async_queue.get()
            if item is None:
                self._async_queue.task_done()
                return
            batch = [item]
            deadline = time.monotonic() + self.async_flush_interval_seconds
            while len(batch) < self.async_batch_size:
                timeout = max(0.0, deadline - time.monotonic())
                try:
                    next_item = self._async_queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if next_item is None:
                    self._async_queue.task_done()
                    self._async_queue.put(None)
                    break
                batch.append(next_item)
            try:
                self._record_batch_with_retries(batch)
            finally:
                for _ in batch:
                    self._async_queue.task_done()

    def _record_batch_with_retries(self, events: List[_QueuedEvent]) -> None:
        try:
            payload = self._batch_payload_from_queue_items(events)
            if payload is not None:
                self._post_payload(payload)
                self._record_delivered(len(events))
        except BaseException as exc:  # pragma: no cover - callback decides handling.
            self._record_failed_batch()
            if self.on_error:
                self.on_error(exc)

    def _batch_payload_from_queue_items(self, queued_events: List[_QueuedEvent]) -> Optional[Dict[str, Any]]:
        payloads = [
            self._event_payload(queued.event, queued.metadata_policy)
            for queued in queued_events
        ]
        if not payloads:
            return None
        if len(payloads) == 1:
            return payloads[0]
        return {"schema_version": SDK_BATCH_SCHEMA_VERSION, "events": payloads}

    def observe(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], T],
        extract_usage: Optional[Callable[[T], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = True,
    ) -> T:
        started = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        agent_fields = _agent_event_fields(agent)
        try:
            result = call()
            completed = datetime.now(timezone.utc)
            extracted = extract_usage(result) if extract_usage else {}
            event = LLMUsageEvent(
                provider=extracted.get("provider") or provider,
                model=extracted.get("model") or model,
                request_id=extracted.get("request_id") or request_id,
                provider_request_id=extracted.get("provider_request_id"),
                trace_id=extracted.get("trace_id") or trace_id,
                status="succeeded",
                input_tokens=extracted.get("input_tokens"),
                output_tokens=extracted.get("output_tokens"),
                total_tokens=extracted.get("total_tokens"),
                reasoning_tokens=extracted.get("reasoning_tokens"),
                cached_input_tokens=extracted.get("cached_input_tokens"),
                extra_usage_units=extracted.get("extra_usage_units") or {},
                cache_hit=extracted.get("cache_hit"),
                started_at=started,
                completed_at=completed,
                latency_ms=int((time.monotonic() - started_monotonic) * 1000),
                **agent_fields,
                attribution=attribution or {},
                metadata={**(metadata or {}), **(extracted.get("metadata") or {})},
            )
            if fire_and_forget:
                self._record_async_with_policy(event, metadata_policy)
            else:
                self._post_payload(self._event_payload(event, metadata_policy))
            return result
        except BaseException as exc:
            completed = datetime.now(timezone.utc)
            event = LLMUsageEvent(
                provider=provider,
                model=model,
                request_id=request_id,
                trace_id=trace_id,
                status="failed",
                started_at=started,
                completed_at=completed,
                latency_ms=int((time.monotonic() - started_monotonic) * 1000),
                error_message=str(exc),
                **agent_fields,
                attribution=attribution or {},
                metadata=metadata or {},
            )
            if fire_and_forget:
                self._record_async_with_policy(event, metadata_policy)
            else:
                self._post_payload(self._event_payload(event, metadata_policy))
            raise

    async def observe_async(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], Any],
        extract_usage: Optional[Callable[[Any], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = True,
    ) -> Any:
        started = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        agent_fields = _agent_event_fields(agent)
        try:
            result = call()
            if inspect.isasyncgen(result):
                raise TypeError("observe_async does not support async generators; use observe_async_stream instead")
            if hasattr(result, "__await__"):
                result = await result
            completed = datetime.now(timezone.utc)
            extracted = extract_usage(result) if extract_usage else {}
            event = LLMUsageEvent(
                provider=extracted.get("provider") or provider,
                model=extracted.get("model") or model,
                request_id=extracted.get("request_id") or request_id,
                provider_request_id=extracted.get("provider_request_id"),
                trace_id=extracted.get("trace_id") or trace_id,
                status="succeeded",
                input_tokens=extracted.get("input_tokens"),
                output_tokens=extracted.get("output_tokens"),
                total_tokens=extracted.get("total_tokens"),
                reasoning_tokens=extracted.get("reasoning_tokens"),
                cached_input_tokens=extracted.get("cached_input_tokens"),
                extra_usage_units=extracted.get("extra_usage_units") or {},
                cache_hit=extracted.get("cache_hit"),
                started_at=started,
                completed_at=completed,
                latency_ms=int((time.monotonic() - started_monotonic) * 1000),
                **agent_fields,
                attribution=attribution or {},
                metadata={**(metadata or {}), **(extracted.get("metadata") or {})},
            )
            if fire_and_forget:
                self._record_async_with_policy(event, metadata_policy)
            else:
                await self._apost_payload(self._event_payload(event, metadata_policy))
            return result
        except BaseException as exc:
            completed = datetime.now(timezone.utc)
            event = LLMUsageEvent(
                provider=provider,
                model=model,
                request_id=request_id,
                trace_id=trace_id,
                status="failed",
                started_at=started,
                completed_at=completed,
                latency_ms=int((time.monotonic() - started_monotonic) * 1000),
                error_message=str(exc),
                **agent_fields,
                attribution=attribution or {},
                metadata=metadata or {},
            )
            if fire_and_forget:
                self._record_async_with_policy(event, metadata_policy)
            else:
                await self._apost_payload(self._event_payload(event, metadata_policy))
            raise

    def observe_stream(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], Iterable[T]],
        extract_usage: Optional[Callable[[List[T]], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = False,
        max_buffered_chunks: int = 256,
    ) -> Iterator[T]:
        started = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        agent_fields = _agent_event_fields(agent)
        chunks = _stream_chunk_buffer(max_buffered_chunks)
        emitted_chunks = 0
        try:
            for chunk in call():
                emitted_chunks += 1
                if extract_usage:
                    chunks.append(chunk)
                yield chunk
            completed = datetime.now(timezone.utc)
            extracted = extract_usage(list(chunks)) if extract_usage else {}
            event = LLMUsageEvent(
                provider=extracted.get("provider") or provider,
                model=extracted.get("model") or model,
                request_id=extracted.get("request_id") or request_id,
                provider_request_id=extracted.get("provider_request_id"),
                trace_id=extracted.get("trace_id") or trace_id,
                status="succeeded",
                input_tokens=extracted.get("input_tokens"),
                output_tokens=extracted.get("output_tokens"),
                total_tokens=extracted.get("total_tokens"),
                reasoning_tokens=extracted.get("reasoning_tokens"),
                cached_input_tokens=extracted.get("cached_input_tokens"),
                extra_usage_units=extracted.get("extra_usage_units") or {},
                cache_hit=extracted.get("cache_hit"),
                started_at=started,
                completed_at=completed,
                latency_ms=int((time.monotonic() - started_monotonic) * 1000),
                **agent_fields,
                attribution=attribution or {},
                metadata={**(metadata or {}), **(extracted.get("metadata") or {}), "streamed": True},
            )
            if fire_and_forget:
                self._record_async_with_policy(event, metadata_policy)
            else:
                self._post_payload(self._event_payload(event, metadata_policy))
        except BaseException as exc:
            completed = datetime.now(timezone.utc)
            event = LLMUsageEvent(
                provider=provider,
                model=model,
                request_id=request_id,
                trace_id=trace_id,
                status="partial" if emitted_chunks > 0 else "failed",
                started_at=started,
                completed_at=completed,
                latency_ms=int((time.monotonic() - started_monotonic) * 1000),
                error_message=str(exc),
                **agent_fields,
                attribution=attribution or {},
                metadata={**(metadata or {}), "streamed": True, "stream_chunks": emitted_chunks},
            )
            if fire_and_forget:
                self._record_async_with_policy(event, metadata_policy)
            else:
                self._post_payload(self._event_payload(event, metadata_policy))
            raise

    async def observe_async_stream(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], Any],
        extract_usage: Optional[Callable[[List[T]], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = False,
        max_buffered_chunks: int = 256,
    ) -> AsyncIterator[T]:
        started = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        agent_fields = _agent_event_fields(agent)
        chunks = _stream_chunk_buffer(max_buffered_chunks)
        emitted_chunks = 0
        try:
            stream = call()
            if hasattr(stream, "__await__"):
                stream = await stream
            if hasattr(stream, "__aiter__"):
                async for chunk in stream:
                    emitted_chunks += 1
                    if extract_usage:
                        chunks.append(chunk)
                    yield chunk
            else:
                for chunk in stream:
                    emitted_chunks += 1
                    if extract_usage:
                        chunks.append(chunk)
                    yield chunk
            completed = datetime.now(timezone.utc)
            extracted = extract_usage(list(chunks)) if extract_usage else {}
            event = LLMUsageEvent(
                provider=extracted.get("provider") or provider,
                model=extracted.get("model") or model,
                request_id=extracted.get("request_id") or request_id,
                provider_request_id=extracted.get("provider_request_id"),
                trace_id=extracted.get("trace_id") or trace_id,
                status="succeeded",
                input_tokens=extracted.get("input_tokens"),
                output_tokens=extracted.get("output_tokens"),
                total_tokens=extracted.get("total_tokens"),
                reasoning_tokens=extracted.get("reasoning_tokens"),
                cached_input_tokens=extracted.get("cached_input_tokens"),
                extra_usage_units=extracted.get("extra_usage_units") or {},
                cache_hit=extracted.get("cache_hit"),
                started_at=started,
                completed_at=completed,
                latency_ms=int((time.monotonic() - started_monotonic) * 1000),
                **agent_fields,
                attribution=attribution or {},
                metadata={**(metadata or {}), **(extracted.get("metadata") or {}), "streamed": True},
            )
            if fire_and_forget:
                self._record_async_with_policy(event, metadata_policy)
            else:
                await self._apost_payload(self._event_payload(event, metadata_policy))
        except BaseException as exc:
            completed = datetime.now(timezone.utc)
            event = LLMUsageEvent(
                provider=provider,
                model=model,
                request_id=request_id,
                trace_id=trace_id,
                status="partial" if emitted_chunks > 0 else "failed",
                started_at=started,
                completed_at=completed,
                latency_ms=int((time.monotonic() - started_monotonic) * 1000),
                error_message=str(exc),
                **agent_fields,
                attribution=attribution or {},
                metadata={**(metadata or {}), "streamed": True, "stream_chunks": emitted_chunks},
            )
            if fire_and_forget:
                self._record_async_with_policy(event, metadata_policy)
            else:
                await self._apost_payload(self._event_payload(event, metadata_policy))
            raise

    def observe_call(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], T],
        extract_usage: Optional[Callable[[T], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = True,
        team_id: Optional[str] = None,
        app_id: Optional[str] = None,
        feature_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        business_unit: Optional[str] = None,
        cost_center: Optional[str] = None,
        product: Optional[str] = None,
        customer_segment: Optional[str] = None,
        end_customer_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        release: Optional[str] = None,
        environment: Optional[str] = None,
        actor_id: Optional[str] = None,
        actor_type: Optional[str] = None,
    ) -> T:
        attribution_overrides = _attribution_overrides(
            team_id=team_id,
            app_id=app_id,
            feature_id=feature_id,
            workflow_id=workflow_id,
            business_unit=business_unit,
            cost_center=cost_center,
            product=product,
            customer_segment=customer_segment,
            end_customer_id=end_customer_id,
            tenant_id=tenant_id,
            release=release,
            environment=environment,
            actor_id=actor_id,
            actor_type=actor_type,
        )
        return self.observe(
            provider=provider,
            model=model,
            call=call,
            extract_usage=extract_usage,
            attribution=_merged_attribution(attribution, attribution_overrides),
            agent=agent,
            metadata=metadata,
            metadata_policy=metadata_policy,
            request_id=request_id,
            trace_id=trace_id,
            fire_and_forget=fire_and_forget,
        )

    async def observe_async_call(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], Any],
        extract_usage: Optional[Callable[[Any], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = True,
        team_id: Optional[str] = None,
        app_id: Optional[str] = None,
        feature_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        business_unit: Optional[str] = None,
        cost_center: Optional[str] = None,
        product: Optional[str] = None,
        customer_segment: Optional[str] = None,
        end_customer_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        release: Optional[str] = None,
        environment: Optional[str] = None,
        actor_id: Optional[str] = None,
        actor_type: Optional[str] = None,
    ) -> Any:
        attribution_overrides = _attribution_overrides(
            team_id=team_id,
            app_id=app_id,
            feature_id=feature_id,
            workflow_id=workflow_id,
            business_unit=business_unit,
            cost_center=cost_center,
            product=product,
            customer_segment=customer_segment,
            end_customer_id=end_customer_id,
            tenant_id=tenant_id,
            release=release,
            environment=environment,
            actor_id=actor_id,
            actor_type=actor_type,
        )
        return await self.observe_async(
            provider=provider,
            model=model,
            call=call,
            extract_usage=extract_usage,
            attribution=_merged_attribution(attribution, attribution_overrides),
            agent=agent,
            metadata=metadata,
            metadata_policy=metadata_policy,
            request_id=request_id,
            trace_id=trace_id,
            fire_and_forget=fire_and_forget,
        )

    def observe_stream_call(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], Iterable[T]],
        extract_usage: Optional[Callable[[List[T]], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = False,
        max_buffered_chunks: int = 256,
        team_id: Optional[str] = None,
        app_id: Optional[str] = None,
        feature_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        business_unit: Optional[str] = None,
        cost_center: Optional[str] = None,
        product: Optional[str] = None,
        customer_segment: Optional[str] = None,
        end_customer_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        release: Optional[str] = None,
        environment: Optional[str] = None,
        actor_id: Optional[str] = None,
        actor_type: Optional[str] = None,
    ) -> Iterator[T]:
        attribution_overrides = _attribution_overrides(
            team_id=team_id,
            app_id=app_id,
            feature_id=feature_id,
            workflow_id=workflow_id,
            business_unit=business_unit,
            cost_center=cost_center,
            product=product,
            customer_segment=customer_segment,
            end_customer_id=end_customer_id,
            tenant_id=tenant_id,
            release=release,
            environment=environment,
            actor_id=actor_id,
            actor_type=actor_type,
        )
        return self.observe_stream(
            provider=provider,
            model=model,
            call=call,
            extract_usage=extract_usage,
            attribution=_merged_attribution(attribution, attribution_overrides),
            agent=agent,
            metadata=metadata,
            metadata_policy=metadata_policy,
            request_id=request_id,
            trace_id=trace_id,
            fire_and_forget=fire_and_forget,
            max_buffered_chunks=max_buffered_chunks,
        )

    async def observe_async_stream_call(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], Any],
        extract_usage: Optional[Callable[[List[T]], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = False,
        max_buffered_chunks: int = 256,
        team_id: Optional[str] = None,
        app_id: Optional[str] = None,
        feature_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        business_unit: Optional[str] = None,
        cost_center: Optional[str] = None,
        product: Optional[str] = None,
        customer_segment: Optional[str] = None,
        end_customer_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        release: Optional[str] = None,
        environment: Optional[str] = None,
        actor_id: Optional[str] = None,
        actor_type: Optional[str] = None,
    ) -> AsyncIterator[T]:
        attribution_overrides = _attribution_overrides(
            team_id=team_id,
            app_id=app_id,
            feature_id=feature_id,
            workflow_id=workflow_id,
            business_unit=business_unit,
            cost_center=cost_center,
            product=product,
            customer_segment=customer_segment,
            end_customer_id=end_customer_id,
            tenant_id=tenant_id,
            release=release,
            environment=environment,
            actor_id=actor_id,
            actor_type=actor_type,
        )
        async for chunk in self.observe_async_stream(
            provider=provider,
            model=model,
            call=call,
            extract_usage=extract_usage,
            attribution=_merged_attribution(attribution, attribution_overrides),
            agent=agent,
            metadata=metadata,
            metadata_policy=metadata_policy,
            request_id=request_id,
            trace_id=trace_id,
            fire_and_forget=fire_and_forget,
            max_buffered_chunks=max_buffered_chunks,
        ):
            yield chunk


class DisabledCloptimaLLMObservability:
    def __init__(self, init_error: Optional[BaseException] = None) -> None:
        self._init_error = init_error

    def is_enabled(self) -> bool:
        return False

    def get_init_error(self) -> Optional[BaseException]:
        return self._init_error

    def run_with_attribution(
        self,
        call: Callable[[], T],
        attribution: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        return run_with_attribution(call, attribution, **kwargs)

    def with_attribution(
        self,
        attribution: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        return with_attribution(attribution, **kwargs)

    def run_with_workflow(
        self,
        call: Callable[[], T],
        name: Optional[str] = None,
        attribution: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        return run_with_workflow(call, name=name, attribution=attribution, **kwargs)

    def with_workflow(
        self,
        name: Optional[str] = None,
        attribution: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        return with_workflow(name=name, attribution=attribution, **kwargs)

    def run_with_task(
        self,
        call: Callable[[], T],
        name: Optional[str] = None,
        attribution: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        return run_with_task(call, name=name, attribution=attribution, **kwargs)

    def with_task(
        self,
        name: Optional[str] = None,
        attribution: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        return with_task(name=name, attribution=attribution, **kwargs)

    def record(self, event: LLMUsageEvent) -> None:
        return None

    def record_batch(self, events: List[LLMUsageEvent]) -> None:
        return None

    def record_async(self, event: LLMUsageEvent) -> None:
        return None

    def stats(self) -> CloptimaLLMClientStats:
        return CloptimaLLMClientStats(
            queued_events=0,
            dropped_events=0,
            delivered_events=0,
            failed_batches=0,
        )

    def flush(self, timeout_seconds: Optional[float] = None) -> bool:
        return True

    def close(self, timeout_seconds: Optional[float] = 5.0) -> bool:
        return True

    async def aclose(self, timeout_seconds: Optional[float] = 5.0) -> bool:
        return True

    def observe(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], T],
        extract_usage: Optional[Callable[[T], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = True,
    ) -> T:
        return call()

    async def observe_async(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], Any],
        extract_usage: Optional[Callable[[Any], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = True,
    ) -> Any:
        result = call()
        if inspect.isasyncgen(result):
            raise TypeError("observe_async does not support async generators; use observe_async_stream instead")
        if hasattr(result, "__await__"):
            result = await result
        return result

    def observe_stream(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], Iterable[T]],
        extract_usage: Optional[Callable[[List[T]], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = False,
        max_buffered_chunks: int = 256,
    ) -> Iterator[T]:
        yield from call()

    async def observe_async_stream(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], Any],
        extract_usage: Optional[Callable[[List[T]], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = False,
        max_buffered_chunks: int = 256,
    ) -> AsyncIterator[T]:
        stream = call()
        if hasattr(stream, "__await__"):
            stream = await stream
        if hasattr(stream, "__aiter__"):
            async for chunk in stream:
                yield chunk
        else:
            for chunk in stream:
                yield chunk

    def observe_call(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], T],
        extract_usage: Optional[Callable[[T], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = True,
        **kwargs: Any,
    ) -> T:
        return self.observe(
            provider=provider,
            model=model,
            call=call,
            extract_usage=extract_usage,
            attribution=_merged_attribution(attribution, _attribution_overrides_from_kwargs(kwargs)),
            agent=agent,
            metadata=metadata,
            metadata_policy=metadata_policy,
            request_id=request_id,
            trace_id=trace_id,
            fire_and_forget=fire_and_forget,
        )

    async def observe_async_call(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], Any],
        extract_usage: Optional[Callable[[Any], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = True,
        **kwargs: Any,
    ) -> Any:
        return await self.observe_async(
            provider=provider,
            model=model,
            call=call,
            extract_usage=extract_usage,
            attribution=_merged_attribution(attribution, _attribution_overrides_from_kwargs(kwargs)),
            agent=agent,
            metadata=metadata,
            metadata_policy=metadata_policy,
            request_id=request_id,
            trace_id=trace_id,
            fire_and_forget=fire_and_forget,
        )

    def observe_stream_call(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], Iterable[T]],
        extract_usage: Optional[Callable[[List[T]], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = False,
        max_buffered_chunks: int = 256,
        **kwargs: Any,
    ) -> Iterator[T]:
        return self.observe_stream(
            provider=provider,
            model=model,
            call=call,
            extract_usage=extract_usage,
            attribution=_merged_attribution(attribution, _attribution_overrides_from_kwargs(kwargs)),
            agent=agent,
            metadata=metadata,
            metadata_policy=metadata_policy,
            request_id=request_id,
            trace_id=trace_id,
            fire_and_forget=fire_and_forget,
            max_buffered_chunks=max_buffered_chunks,
        )

    async def observe_async_stream_call(
        self,
        *,
        provider: str,
        model: str,
        call: Callable[[], Any],
        extract_usage: Optional[Callable[[List[T]], Dict[str, Any]]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        agent: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        fire_and_forget: bool = False,
        max_buffered_chunks: int = 256,
        **kwargs: Any,
    ) -> AsyncIterator[T]:
        async for chunk in self.observe_async_stream(
            provider=provider,
            model=model,
            call=call,
            extract_usage=extract_usage,
            attribution=_merged_attribution(attribution, _attribution_overrides_from_kwargs(kwargs)),
            agent=agent,
            metadata=metadata,
            metadata_policy=metadata_policy,
            request_id=request_id,
            trace_id=trace_id,
            fire_and_forget=fire_and_forget,
            max_buffered_chunks=max_buffered_chunks,
        ):
            yield chunk


def disabled_client(init_error: Optional[BaseException] = None) -> DisabledCloptimaLLMObservability:
    return DisabledCloptimaLLMObservability(init_error=init_error)


def is_enabled(
    *,
    env: Optional[Mapping[str, str]] = None,
    enabled: Optional[bool] = None,
    api_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    default_attribution: Optional[Union[LLMAttribution, Dict[str, Any]]] = None,
) -> bool:
    current_env = _current_env(env)
    enabled_flag = enabled if enabled is not None else _clean_bool(current_env.get(INIT_ENABLED_ENV))
    if enabled_flag is False:
        return False

    attribution = dict(default_attribution or {})
    if isinstance(default_attribution, LLMAttribution):
        attribution = default_attribution.__dict__.copy()
    app_id = _clean_str(attribution.get("app_id")) or _clean_str(current_env.get(INIT_APP_ID_ENV))
    environment = _clean_str(attribution.get("environment")) or _clean_str(current_env.get(INIT_ENVIRONMENT_ENV)) or "production"
    resolved_api_key = _clean_str(api_key) or _clean_str(current_env.get(INIT_API_KEY_ENV))
    return bool(resolved_api_key and app_id and environment)


def init_from_env(
    *,
    env: Optional[Mapping[str, str]] = None,
    enabled: Optional[bool] = None,
    strict: bool = False,
    on_init_error: Optional[Callable[[BaseException], None]] = None,
    api_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    default_attribution: Optional[Union[LLMAttribution, Dict[str, Any]]] = None,
    delivery_mode: Optional[str] = None,
    otlp_headers: Optional[Dict[str, str]] = None,
    otlp_service_name: Optional[str] = None,
    otlp_service_version: Optional[str] = None,
    sdk_name: str = "cloptima-llm-observability",
    sdk_version: Optional[str] = None,
    timeout_seconds: float = 3.0,
    metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
    on_error: Optional[Callable[[BaseException], None]] = None,
    on_drop: Optional[Callable[[LLMUsageEvent, str], None]] = None,
    async_queue_max_size: int = 1000,
    async_batch_size: int = 20,
    async_flush_interval_seconds: float = 0.25,
    async_retry_count: int = 2,
    async_retry_backoff_seconds: float = 0.1,
    async_retry_jitter_ratio: float = 0.2,
    async_http_client: Optional[Any] = None,
) -> Union[CloptimaLLMObservability, DisabledCloptimaLLMObservability]:
    current_env = _current_env(env)
    enabled_flag = enabled if enabled is not None else _clean_bool(current_env.get(INIT_ENABLED_ENV))
    if enabled_flag is False:
        return disabled_client()

    attribution_values = dict(default_attribution or {})
    if isinstance(default_attribution, LLMAttribution):
        attribution_values = default_attribution.__dict__.copy()

    resolved_api_base_url = _clean_str(api_base_url) or _clean_str(current_env.get(INIT_API_BASE_URL_ENV)) or DEFAULT_API_BASE_URL
    resolved_api_key = _clean_str(api_key) or _clean_str(current_env.get(INIT_API_KEY_ENV))
    resolved_app_id = _clean_str(attribution_values.get("app_id")) or _clean_str(current_env.get(INIT_APP_ID_ENV))
    resolved_environment = _clean_str(attribution_values.get("environment")) or _clean_str(current_env.get(INIT_ENVIRONMENT_ENV)) or "production"
    resolved_team_id = _clean_str(attribution_values.get("team_id")) or _clean_str(current_env.get(INIT_TEAM_ID_ENV))

    missing_fields = [
        INIT_API_KEY_ENV if not resolved_api_key else None,
        INIT_APP_ID_ENV if not resolved_app_id else None,
    ]
    missing = [field for field in missing_fields if field]
    if missing:
        if enabled_flag is True:
            error = RuntimeError(
                "Cloptima LLM observability is enabled but missing required configuration: "
                + ", ".join(missing)
            )
            if on_init_error:
                on_init_error(error)
            if strict:
                raise error
            return disabled_client(error)
        return disabled_client()

    return CloptimaLLMObservability(
        api_base_url=resolved_api_base_url or "",
        api_key=resolved_api_key or "",
        default_attribution=LLMAttribution(
            app_id=resolved_app_id or "",
            environment=resolved_environment or "",
            team_id=resolved_team_id,
            feature_id=_clean_str(attribution_values.get("feature_id")),
            workflow_id=_clean_str(attribution_values.get("workflow_id")),
            business_unit=_clean_str(attribution_values.get("business_unit")),
            cost_center=_clean_str(attribution_values.get("cost_center")),
            product=_clean_str(attribution_values.get("product")),
            customer_segment=_clean_str(attribution_values.get("customer_segment")),
            end_customer_id=_clean_str(attribution_values.get("end_customer_id")),
            tenant_id=_clean_str(attribution_values.get("tenant_id")),
            release=_clean_str(attribution_values.get("release")),
            actor_id=_clean_str(attribution_values.get("actor_id")),
            actor_type=_clean_str(attribution_values.get("actor_type")),
        ),
        delivery_mode=delivery_mode or _clean_str(current_env.get(INIT_DELIVERY_MODE_ENV)) or "cloptima_http",
        otlp_headers=otlp_headers,
        otlp_service_name=otlp_service_name or _clean_str(current_env.get(INIT_OTLP_SERVICE_NAME_ENV)) or "cloptima-llm-observability",
        otlp_service_version=otlp_service_version or _clean_str(current_env.get(INIT_OTLP_SERVICE_VERSION_ENV)),
        sdk_name=sdk_name,
        sdk_version=sdk_version,
        timeout_seconds=timeout_seconds,
        metadata_policy=metadata_policy,
        on_error=on_error,
        on_drop=on_drop,
        async_queue_max_size=async_queue_max_size,
        async_batch_size=async_batch_size,
        async_flush_interval_seconds=async_flush_interval_seconds,
        async_retry_count=async_retry_count,
        async_retry_backoff_seconds=async_retry_backoff_seconds,
        async_retry_jitter_ratio=async_retry_jitter_ratio,
        async_http_client=async_http_client,
    )


def extract_openai_usage(response: Any) -> Dict[str, Any]:
    record = _coerce_mapping(response)
    usage = _nested_mapping(record, "usage")
    prompt_details = _nested_mapping(usage, "prompt_tokens_details", "promptTokensDetails")
    completion_details = _nested_mapping(usage, "completion_tokens_details", "completionTokensDetails")
    cached_tokens = _clean_int(prompt_details.get("cached_tokens"))
    return _strip_none(
        {
            "provider": "openai",
            "provider_request_id": _mapping_field(record, "id"),
            "model": _mapping_field(record, "model"),
            "input_tokens": _clean_int(usage.get("prompt_tokens")),
            "output_tokens": _clean_int(usage.get("completion_tokens")),
            "total_tokens": _clean_int(usage.get("total_tokens")),
            "reasoning_tokens": _clean_int(completion_details.get("reasoning_tokens")),
            "cached_input_tokens": cached_tokens,
            "extra_usage_units": _clean_usage_map(
                {
                    "cache_write": prompt_details.get("cache_creation_input_tokens"),
                    "cache_write_5m": prompt_details.get("cache_creation_input_tokens_5m"),
                    "cache_write_1h": prompt_details.get("cache_creation_input_tokens_1h"),
                }
            ) or None,
            "cache_hit": True if cached_tokens else None,
        }
    )


def extract_openai_stream_usage(chunks: Iterable[Any]) -> Dict[str, Any]:
    last_with_usage: Dict[str, Any] = {}
    last_id = None
    last_model = None
    for chunk in chunks:
        record = _coerce_mapping(chunk)
        if not record:
            continue
        last_id = _mapping_field(record, "id") or last_id
        last_model = _mapping_field(record, "model") or last_model
        if _nested_mapping(record, "usage"):
            last_with_usage = record
    if not last_with_usage:
        return _strip_none({"provider_request_id": last_id, "model": last_model})
    extracted = extract_openai_usage(last_with_usage)
    if not extracted.get("provider_request_id"):
        extracted["provider_request_id"] = last_id
    if not extracted.get("model"):
        extracted["model"] = last_model
    return _strip_none(extracted)


def extract_azure_openai_usage(response: Any) -> Dict[str, Any]:
    record = _coerce_mapping(response)
    extracted = extract_openai_usage(response)
    extracted["provider"] = "azure_openai"
    if not extracted.get("model"):
        extracted["model"] = _mapping_field(record, "deployment_name", "deployment", "model")
    return _strip_none(extracted)


def extract_anthropic_usage(response: Any) -> Dict[str, Any]:
    record = _coerce_mapping(response)
    usage = _nested_mapping(record, "usage")
    input_tokens = _clean_int(usage.get("input_tokens"))
    output_tokens = _clean_int(usage.get("output_tokens"))
    cache_read_tokens = _clean_int(usage.get("cache_read_input_tokens"))
    extra_usage_units = _clean_usage_map(
        {
            "cache_write": usage.get("cache_creation_input_tokens"),
            "cache_write_5m": usage.get("cache_creation_input_tokens_5m"),
            "cache_write_1h": usage.get("cache_creation_input_tokens_1h"),
            "server_tool_use": usage.get("server_tool_use"),
        }
    )
    return _strip_none(
        {
            "provider": "anthropic",
            "provider_request_id": _mapping_field(record, "id"),
            "model": _mapping_field(record, "model"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": _clean_int(usage.get("total_tokens")) or (
                (input_tokens or 0) + (output_tokens or 0)
                if input_tokens is not None or output_tokens is not None
                else None
            ),
            "cached_input_tokens": cache_read_tokens,
            "extra_usage_units": extra_usage_units or None,
            "cache_hit": True if cache_read_tokens else None,
        }
    )


def extract_anthropic_stream_usage(chunks: Iterable[Any]) -> Dict[str, Any]:
    message_id = None
    model = None
    input_tokens = None
    output_tokens = None
    cache_read_tokens = None
    cache_write_tokens = None
    for chunk in chunks:
        record = _coerce_mapping(chunk)
        if not record:
            continue
        message = _nested_mapping(record, "message")
        if message:
            message_id = _mapping_field(message, "id") or message_id
            model = _mapping_field(message, "model") or model
            usage = _nested_mapping(message, "usage")
        else:
            usage = _nested_mapping(record, "usage")
        input_tokens = _clean_int(usage.get("input_tokens")) or input_tokens
        output_tokens = _clean_int(usage.get("output_tokens")) or output_tokens
        cache_read_tokens = _clean_int(usage.get("cache_read_input_tokens")) or cache_read_tokens
        cache_write_tokens = _clean_int(usage.get("cache_creation_input_tokens")) or cache_write_tokens
    total_tokens = None
    if input_tokens is not None or output_tokens is not None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return _strip_none(
        {
            "provider": "anthropic",
            "provider_request_id": message_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": cache_read_tokens,
            "extra_usage_units": _clean_usage_map({"cache_write": cache_write_tokens}) or None,
            "cache_hit": True if cache_read_tokens else None,
        }
    )


def extract_gemini_usage(response: Any) -> Dict[str, Any]:
    record = _coerce_mapping(response)
    usage = _nested_mapping(record, "usageMetadata", "usage_metadata")
    cached_tokens = _clean_int(_mapping_field(usage, "cachedContentTokenCount", "cached_content_token_count"))
    return _strip_none(
        {
            "provider": _mapping_field(record, "provider") or "gemini",
            "provider_request_id": _mapping_field(record, "responseId", "response_id", "id", "name"),
            "model": _mapping_field(record, "modelVersion", "model_version", "model"),
            "input_tokens": _clean_int(_mapping_field(usage, "promptTokenCount", "prompt_token_count", "inputTokenCount", "input_token_count")),
            "output_tokens": _clean_int(_mapping_field(usage, "responseTokenCount", "response_token_count", "candidatesTokenCount", "candidates_token_count", "outputTokenCount", "output_token_count")),
            "total_tokens": _clean_int(_mapping_field(usage, "totalTokenCount", "total_token_count")),
            "reasoning_tokens": _clean_int(_mapping_field(usage, "thoughtsTokenCount", "thoughts_token_count", "reasoningTokenCount", "reasoning_token_count")),
            "cached_input_tokens": cached_tokens,
            "cache_hit": True if cached_tokens else None,
        }
    )


def extract_vertex_usage(response: Any) -> Dict[str, Any]:
    extracted = extract_gemini_usage(response)
    extracted["provider"] = "vertex_ai"
    return _strip_none(extracted)


def extract_gemini_stream_usage(chunks: Iterable[Any]) -> Dict[str, Any]:
    last_with_usage: Dict[str, Any] = {}
    last_id = None
    last_model = None
    for chunk in chunks:
        record = _coerce_mapping(chunk)
        if not record:
            continue
        last_id = _mapping_field(record, "responseId", "response_id", "id", "name") or last_id
        last_model = _mapping_field(record, "modelVersion", "model_version", "model") or last_model
        if _nested_mapping(record, "usageMetadata", "usage_metadata"):
            last_with_usage = record
    if not last_with_usage:
        return _strip_none({"provider": "gemini", "provider_request_id": last_id, "model": last_model})
    extracted = extract_gemini_usage(last_with_usage)
    if not extracted.get("provider_request_id"):
        extracted["provider_request_id"] = last_id
    if not extracted.get("model"):
        extracted["model"] = last_model
    return _strip_none(extracted)


def extract_vertex_stream_usage(chunks: Iterable[Any]) -> Dict[str, Any]:
    extracted = extract_gemini_stream_usage(chunks)
    extracted["provider"] = "vertex_ai"
    return _strip_none(extracted)


def extract_bedrock_usage(response: Any) -> Dict[str, Any]:
    record = _coerce_mapping(response)
    usage = _nested_mapping(record, "usage")
    metrics = _nested_mapping(record, "metrics")
    metadata = _nested_mapping(record, "ResponseMetadata")
    return _strip_none(
        {
            "provider": "bedrock",
            "provider_request_id": _mapping_field(record, "requestId", "request_id") or _mapping_field(metadata, "RequestId"),
            "model": _mapping_field(record, "modelId", "model_id", "model"),
            "input_tokens": _clean_int(_mapping_field(usage, "inputTokens", "input_tokens")),
            "output_tokens": _clean_int(_mapping_field(usage, "outputTokens", "output_tokens")),
            "total_tokens": _clean_int(_mapping_field(usage, "totalTokens", "total_tokens")),
            "latency_ms": _clean_int(_mapping_field(metrics, "latencyMs", "latency_ms")),
        }
    )


def extract_bedrock_stream_usage(chunks: Iterable[Any]) -> Dict[str, Any]:
    """Aggregate Amazon Bedrock streaming usage deltas across chunks."""
    request_id = None
    model = None
    input_tokens = 0
    output_tokens = 0
    total_tokens = None
    saw_usage = False
    for chunk in chunks:
        record = _coerce_mapping(chunk)
        if not record:
            continue
        request_id = _mapping_field(record, "requestId", "request_id") or request_id
        model = _mapping_field(record, "modelId", "model_id", "model") or model
        usage = _nested_mapping(record, "usage")
        if not usage:
            continue
        input_count = _clean_int(_mapping_field(usage, "inputTokens", "input_tokens"))
        output_count = _clean_int(_mapping_field(usage, "outputTokens", "output_tokens"))
        total_count = _clean_int(_mapping_field(usage, "totalTokens", "total_tokens"))
        if input_count is not None:
            input_tokens += input_count
            saw_usage = True
        if output_count is not None:
            output_tokens += output_count
            saw_usage = True
        total_tokens = total_count if total_count is not None else total_tokens
    return _strip_none(
        {
            "provider": "bedrock",
            "provider_request_id": request_id,
            "model": model,
            "input_tokens": input_tokens if saw_usage else None,
            "output_tokens": output_tokens if saw_usage else None,
            "total_tokens": total_tokens if total_tokens is not None else (input_tokens + output_tokens if saw_usage else None),
        }
    )


PROVIDER_USAGE_EXTRACTORS = (
    MappingProxyType(
        {
            "provider": "openai",
            "aliases": ("openai",),
            "response_extractor": extract_openai_usage,
            "stream_extractor": extract_openai_stream_usage,
        }
    ),
    MappingProxyType(
        {
            "provider": "azure_openai",
            "aliases": ("azure_openai", "azure-openai", "azure"),
            "response_extractor": extract_azure_openai_usage,
            "stream_extractor": lambda chunks: _strip_none(
                {
                    **extract_openai_stream_usage(chunks),
                    "provider": "azure_openai",
                }
            ),
        }
    ),
    MappingProxyType(
        {
            "provider": "anthropic",
            "aliases": ("anthropic",),
            "response_extractor": extract_anthropic_usage,
            "stream_extractor": extract_anthropic_stream_usage,
        }
    ),
    MappingProxyType(
        {
            "provider": "gemini",
            "aliases": ("gemini",),
            "response_extractor": extract_gemini_usage,
            "stream_extractor": extract_gemini_stream_usage,
        }
    ),
    MappingProxyType(
        {
            "provider": "vertex_ai",
            "aliases": ("vertex_ai", "vertex-ai", "vertex"),
            "response_extractor": extract_vertex_usage,
            "stream_extractor": extract_vertex_stream_usage,
        }
    ),
    MappingProxyType(
        {
            "provider": "bedrock",
            "aliases": ("bedrock",),
            "response_extractor": extract_bedrock_usage,
            "stream_extractor": extract_bedrock_stream_usage,
        }
    ),
)


PROVIDER_SUPPORT_MATRIX = tuple(
    MappingProxyType(
        {
            "provider": descriptor["provider"],
            "aliases": tuple(descriptor["aliases"]),
            "response": True,
            "stream": descriptor.get("stream_extractor") is not None,
        }
    )
    for descriptor in PROVIDER_USAGE_EXTRACTORS
)


def get_provider_usage_extractor(provider: Optional[str]) -> Optional[UsageExtractor]:
    normalized = _clean_str(provider or "")
    if not normalized:
        return None
    lowered = normalized.lower()
    for descriptor in PROVIDER_USAGE_EXTRACTORS:
        if lowered in descriptor["aliases"]:
            return descriptor["response_extractor"]
    return None


def get_provider_stream_usage_extractor(provider: Optional[str]) -> Optional[UsageExtractor]:
    normalized = _clean_str(provider or "")
    if not normalized:
        return None
    lowered = normalized.lower()
    for descriptor in PROVIDER_USAGE_EXTRACTORS:
        if lowered in descriptor["aliases"]:
            return descriptor.get("stream_extractor")
    return None


def list_supported_providers() -> List[Dict[str, Any]]:
    return [
        {
            "provider": descriptor["provider"],
            "aliases": list(descriptor["aliases"]),
            "response": descriptor["response"],
            "stream": descriptor["stream"],
        }
        for descriptor in PROVIDER_SUPPORT_MATRIX
    ]


def _normalize_openai_compatible_provider(provider: Optional[str]) -> str:
    normalized = _clean_str(provider or "openai")
    if not normalized:
        return "openai"
    lowered = normalized.lower()
    if lowered in {"azure", "azure-openai", "azure_openai"}:
        return "azure_openai"
    return "openai"


def _select_openai_compatible_response_extractor(provider: Optional[str]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    return extract_azure_openai_usage if _normalize_openai_compatible_provider(provider) == "azure_openai" else extract_openai_usage


def _select_openai_compatible_stream_extractor(provider: Optional[str]) -> Callable[[Iterable[Dict[str, Any]]], Dict[str, Any]]:
    if _normalize_openai_compatible_provider(provider) == "azure_openai":
        def _extractor(chunks: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
            extracted = extract_openai_stream_usage(chunks)
            extracted["provider"] = "azure_openai"
            return extracted
        return _extractor
    return extract_openai_stream_usage


def instrument_openai_compatible_response(
    client: CloptimaLLMObservability,
    response: Any,
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    attribution: Optional[Dict[str, Any]] = None,
    agent: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    fire_and_forget: bool = True,
) -> Dict[str, Any]:
    normalized_provider = _normalize_openai_compatible_provider(provider)
    extractor = _select_openai_compatible_response_extractor(normalized_provider)
    started = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    resolved = response() if callable(response) else response
    completed = datetime.now(timezone.utc)
    extracted = extractor(resolved)
    event = LLMUsageEvent(
        provider=extracted.get("provider") or normalized_provider,
        model=extracted.get("model") or model or _clean_str(resolved.get("model")) or "unknown",
        request_id=extracted.get("request_id") or request_id,
        provider_request_id=extracted.get("provider_request_id"),
        trace_id=extracted.get("trace_id") or trace_id,
        status="succeeded",
        input_tokens=extracted.get("input_tokens"),
        output_tokens=extracted.get("output_tokens"),
        total_tokens=extracted.get("total_tokens"),
        reasoning_tokens=extracted.get("reasoning_tokens"),
        cached_input_tokens=extracted.get("cached_input_tokens"),
        extra_usage_units=extracted.get("extra_usage_units") or {},
        cache_hit=extracted.get("cache_hit"),
        started_at=started if callable(response) else None,
        completed_at=completed if callable(response) else None,
        latency_ms=int((time.monotonic() - started_monotonic) * 1000) if callable(response) else None,
        **_agent_event_fields(agent),
        attribution=attribution or {},
        metadata={**(metadata or {}), **(extracted.get("metadata") or {})},
    )
    if fire_and_forget:
        client.record_async(event)
    else:
        client.record(event)
    return resolved


async def ainstrument_openai_compatible_response(
    client: CloptimaLLMObservability,
    response: Any,
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    attribution: Optional[Dict[str, Any]] = None,
    agent: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    fire_and_forget: bool = True,
) -> Dict[str, Any]:
    normalized_provider = _normalize_openai_compatible_provider(provider)
    extractor = _select_openai_compatible_response_extractor(normalized_provider)
    started = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    resolved = response() if callable(response) else response
    if hasattr(resolved, "__await__"):
        resolved = await resolved
        completed = datetime.now(timezone.utc)
        started_at = started
        completed_at = completed
        latency_ms = int((time.monotonic() - started_monotonic) * 1000)
    else:
        completed = datetime.now(timezone.utc)
        started_at = started if callable(response) else None
        completed_at = completed if callable(response) else None
        latency_ms = int((time.monotonic() - started_monotonic) * 1000) if callable(response) else None
    extracted = extractor(resolved)
    event = LLMUsageEvent(
        provider=extracted.get("provider") or normalized_provider,
        model=extracted.get("model") or model or _clean_str(resolved.get("model")) or "unknown",
        request_id=extracted.get("request_id") or request_id,
        provider_request_id=extracted.get("provider_request_id"),
        trace_id=extracted.get("trace_id") or trace_id,
        status="succeeded",
        input_tokens=extracted.get("input_tokens"),
        output_tokens=extracted.get("output_tokens"),
        total_tokens=extracted.get("total_tokens"),
        reasoning_tokens=extracted.get("reasoning_tokens"),
        cached_input_tokens=extracted.get("cached_input_tokens"),
        extra_usage_units=extracted.get("extra_usage_units") or {},
        cache_hit=extracted.get("cache_hit"),
        started_at=started_at,
        completed_at=completed_at,
        latency_ms=latency_ms,
        **_agent_event_fields(agent),
        attribution=attribution or {},
        metadata={**(metadata or {}), **(extracted.get("metadata") or {})},
    )
    if fire_and_forget:
        client.record_async(event)
    else:
        await client.arecord(event)
    return resolved


def instrument_openai_compatible_stream(
    client: CloptimaLLMObservability,
    stream: Iterable[Dict[str, Any]],
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    attribution: Optional[Dict[str, Any]] = None,
    agent: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    fire_and_forget: bool = False,
    max_buffered_chunks: int = 256,
) -> Iterator[Dict[str, Any]]:
    normalized_provider = _normalize_openai_compatible_provider(provider)
    return client.observe_stream(
        provider=normalized_provider,
        model=model or "unknown",
        call=lambda: stream,
        extract_usage=_select_openai_compatible_stream_extractor(normalized_provider),
        attribution=attribution,
        agent=agent,
        metadata=metadata,
        request_id=request_id,
        trace_id=trace_id,
        fire_and_forget=fire_and_forget,
        max_buffered_chunks=max_buffered_chunks,
    )


def ainstrument_openai_compatible_stream(
    client: CloptimaLLMObservability,
    stream: Any,
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    attribution: Optional[Dict[str, Any]] = None,
    agent: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    fire_and_forget: bool = False,
    max_buffered_chunks: int = 256,
) -> AsyncIterator[Dict[str, Any]]:
    normalized_provider = _normalize_openai_compatible_provider(provider)
    return client.observe_async_stream(
        provider=normalized_provider,
        model=model or "unknown",
        call=lambda: stream,
        extract_usage=_select_openai_compatible_stream_extractor(normalized_provider),
        attribution=attribution,
        agent=agent,
        metadata=metadata,
        request_id=request_id,
        trace_id=trace_id,
        fire_and_forget=fire_and_forget,
        max_buffered_chunks=max_buffered_chunks,
    )


def _header_dict(headers: Any) -> Dict[str, str]:
    if headers is None:
        return {}
    if isinstance(headers, dict):
        return {
            str(key).lower(): ", ".join(map(str, value)) if isinstance(value, list) else str(value)
            for key, value in headers.items()
            if value is not None
        }
    if hasattr(headers, "items"):
        return {
            str(key).lower(): str(value)
            for key, value in headers.items()
            if value is not None
        }
    return {}


def _selected_header_metadata(headers: Dict[str, str], include_headers: Optional[List[str]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for header in include_headers or []:
        normalized = str(header or "").strip().lower()
        if normalized and normalized in headers:
            result[f"http_header_{normalized.replace('-', '_')}"] = headers[normalized]
    return result


def _request_context_request_id(headers: Dict[str, str], request_id_header: Optional[str]) -> Optional[str]:
    return _clean_str(headers.get((request_id_header or "x-request-id").lower()))


def _request_context_trace_id(headers: Dict[str, str], trace_id_header: Optional[str]) -> Optional[str]:
    return _clean_str(headers.get((trace_id_header or "x-trace-id").lower())) or _clean_str(headers.get("traceparent"))


def _upper_clean_str(value: Any) -> Optional[str]:
    cleaned = _clean_str(value)
    return cleaned.upper() if cleaned else None


def instrument_fastapi_request_context(
    request: Any,
    *,
    attribution: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    include_headers: Optional[List[str]] = None,
    request_id_header: Optional[str] = None,
    trace_id_header: Optional[str] = None,
    route: Optional[str] = None,
) -> Dict[str, Any]:
    headers = _header_dict(getattr(request, "headers", None))
    url = getattr(request, "url", None)
    path = _clean_str(route) or _clean_str(getattr(url, "path", None))
    client = getattr(request, "client", None)
    return {
        "attribution": attribution or {},
        "request_id": _request_context_request_id(headers, request_id_header),
        "trace_id": _request_context_trace_id(headers, trace_id_header),
        "metadata": _strip_none({
            **(metadata or {}),
            "http_method": _upper_clean_str(getattr(request, "method", None) or getattr(getattr(request, "scope", {}), "get", lambda *_: None)("method")),
            "http_route": path,
            "http_path": path,
            "http_host": _clean_str(getattr(url, "netloc", None)) or _clean_str(headers.get("host")),
            "client_ip": _clean_str(getattr(client, "host", None)),
            "user_agent": _clean_str(headers.get("user-agent")),
            **_selected_header_metadata(headers, include_headers),
        }),
    }


def instrument_flask_request_context(
    request: Any,
    *,
    attribution: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    include_headers: Optional[List[str]] = None,
    request_id_header: Optional[str] = None,
    trace_id_header: Optional[str] = None,
    route: Optional[str] = None,
) -> Dict[str, Any]:
    headers = _header_dict(getattr(request, "headers", None))
    path = _clean_str(route) or _clean_str(getattr(request, "url_rule", None).rule if getattr(request, "url_rule", None) else None) or _clean_str(getattr(request, "path", None))
    return {
        "attribution": attribution or {},
        "request_id": _request_context_request_id(headers, request_id_header),
        "trace_id": _request_context_trace_id(headers, trace_id_header),
        "metadata": _strip_none({
            **(metadata or {}),
            "http_method": _upper_clean_str(getattr(request, "method", None)),
            "http_route": path,
            "http_path": _clean_str(getattr(request, "path", None)) or path,
            "http_host": _clean_str(getattr(request, "host", None)) or _clean_str(headers.get("host")),
            "client_ip": _clean_str(getattr(request, "remote_addr", None)),
            "user_agent": _clean_str(headers.get("user-agent")),
            **_selected_header_metadata(headers, include_headers),
        }),
    }


def instrument_httpx_transport_metadata(
    request: Any,
    *,
    attribution: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    include_headers: Optional[List[str]] = None,
    request_id_header: Optional[str] = None,
    trace_id_header: Optional[str] = None,
) -> Dict[str, Any]:
    headers = _header_dict(getattr(request, "headers", None))
    url = getattr(request, "url", None)
    return {
        "attribution": attribution or {},
        "request_id": _request_context_request_id(headers, request_id_header),
        "trace_id": _request_context_trace_id(headers, trace_id_header),
        "metadata": _strip_none({
            **(metadata or {}),
            "http_method": _upper_clean_str(getattr(request, "method", None)),
            "http_host": _clean_str(getattr(url, "host", None)) or _clean_str(headers.get("host")),
            "http_path": _clean_str(getattr(url, "path", None)),
            "http_route": _clean_str(getattr(url, "path", None)),
            "provider_endpoint": _clean_str(str(url)) if url is not None else None,
            "user_agent": _clean_str(headers.get("user-agent")),
            **_selected_header_metadata(headers, include_headers),
        }),
    }


def _select_provider_response_extractor(provider: Optional[str]) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    return get_provider_usage_extractor(provider)


def _httpx_response_json(response: Any) -> Tuple[Optional[Dict[str, Any]], bool]:
    if isinstance(response, dict):
        return response, True
    json_method = getattr(response, "json", None)
    if not callable(json_method):
        return None, False
    try:
        payload = json_method()
    except BaseException:
        return None, False
    return (payload, isinstance(payload, dict))


def _httpx_response_status_code(response: Any) -> Optional[int]:
    return _clean_int(getattr(response, "status_code", None) or getattr(response, "status", None))


def _httpx_provider_request_id(response: Any) -> Optional[str]:
    headers = _header_dict(getattr(response, "headers", None))
    for header_name in (
        "openai-request-id",
        "anthropic-request-id",
        "x-request-id",
        "request-id",
        "x-amzn-requestid",
        "x-amz-request-id",
    ):
        if headers.get(header_name):
            return _clean_str(headers.get(header_name))
    return None


class _HttpxTransportUrlProxy:
    def __init__(self, raw_url: Any) -> None:
        value = str(raw_url)
        parsed = urlparse(value)
        self.host = parsed.hostname or parsed.netloc or None
        self.path = parsed.path or "/"
        self._raw_url = value

    def __str__(self) -> str:
        return self._raw_url


class _HttpxRequestProxy:
    def __init__(self, method: Any, url: Any, headers: Any) -> None:
        self.method = method
        self.url = _HttpxTransportUrlProxy(url)
        self.headers = headers


def _resolve_httpx_instrumentation_options(
    request: Any,
    *,
    provider: Optional[str],
    model: Optional[str],
    attribution: Optional[Dict[str, Any]],
    agent: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
    metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]],
    request_id: Optional[str],
    trace_id: Optional[str],
    fire_and_forget: bool,
    include_headers: Optional[List[str]],
    request_id_header: Optional[str],
    trace_id_header: Optional[str],
    resolve_options: Optional[Callable[[Any], Optional[Dict[str, Any]]]],
) -> Dict[str, Any]:
    resolved = resolve_options(request) if resolve_options else None
    resolved = resolved if isinstance(resolved, dict) else {}
    request_context = instrument_httpx_transport_metadata(
        request,
        include_headers=include_headers,
        request_id_header=request_id_header,
        trace_id_header=trace_id_header,
    )
    resolved_attribution = resolved.get("attribution") if isinstance(resolved.get("attribution"), dict) else {}
    resolved_agent = resolved.get("agent") if isinstance(resolved.get("agent"), dict) else {}
    resolved_metadata = resolved.get("metadata") if isinstance(resolved.get("metadata"), dict) else {}
    return {
        "provider": _clean_str(resolved.get("provider")) or _clean_str(provider),
        "model": _clean_str(resolved.get("model")) or _clean_str(model) or "unknown",
        "attribution": _merged_attribution(attribution, resolved_attribution),
        "agent": _strip_none({**(agent or {}), **resolved_agent}),
        "metadata": _strip_none({**request_context.get("metadata", {}), **(metadata or {}), **resolved_metadata}),
        "metadata_policy": resolved.get("metadata_policy") if resolved.get("metadata_policy") is not None else metadata_policy,
        "request_id": _clean_str(resolved.get("request_id")) or _clean_str(request_id) or _clean_str(request_context.get("request_id")),
        "trace_id": _clean_str(resolved.get("trace_id")) or _clean_str(trace_id) or _clean_str(request_context.get("trace_id")),
        "fire_and_forget": resolved.get("fire_and_forget") if isinstance(resolved.get("fire_and_forget"), bool) else fire_and_forget,
    }


def _extract_httpx_usage_fields(
    response: Any,
    *,
    provider: Optional[str],
    extract_usage: Optional[Callable[[Any], Dict[str, Any]]],
    on_instrumentation_error: Optional[Callable[[BaseException], None]],
) -> Tuple[Dict[str, Any], bool]:
    if extract_usage is not None:
        try:
            extracted = extract_usage(response) or {}
        except BaseException as exc:
            if on_instrumentation_error:
                on_instrumentation_error(exc)
            return {}, False
        return extracted if isinstance(extracted, dict) else {}, False

    payload, parsed = _httpx_response_json(response)
    extractor = _select_provider_response_extractor(provider)
    if not parsed or payload is None or extractor is None:
        return {}, parsed
    try:
        extracted = extractor(payload) or {}
    except BaseException as exc:
        if on_instrumentation_error:
            on_instrumentation_error(exc)
        return {}, parsed
    return extracted if isinstance(extracted, dict) else {}, parsed


def _build_httpx_usage_event(
    response: Any,
    *,
    options: Dict[str, Any],
    extract_usage: Optional[Callable[[Any], Dict[str, Any]]],
    on_instrumentation_error: Optional[Callable[[BaseException], None]],
    started: datetime,
    started_monotonic: float,
    completed: datetime,
) -> LLMUsageEvent:
    extracted, response_json_parsed = _extract_httpx_usage_fields(
        response,
        provider=options.get("provider"),
        extract_usage=extract_usage,
        on_instrumentation_error=on_instrumentation_error,
    )
    status_code = _httpx_response_status_code(response)
    metadata = _strip_none(
        {
            **(options.get("metadata") or {}),
            "http_status_code": status_code,
            "response_json_parsed": response_json_parsed,
            **(extracted.get("metadata") or {}),
        }
    )
    return LLMUsageEvent(
        provider=extracted.get("provider") or options.get("provider") or "openai",
        model=extracted.get("model") or options.get("model") or "unknown",
        request_id=extracted.get("request_id") or options.get("request_id"),
        provider_request_id=extracted.get("provider_request_id") or _httpx_provider_request_id(response),
        trace_id=extracted.get("trace_id") or options.get("trace_id"),
        status="failed" if status_code is not None and status_code >= 400 else "succeeded",
        input_tokens=extracted.get("input_tokens"),
        output_tokens=extracted.get("output_tokens"),
        total_tokens=extracted.get("total_tokens"),
        reasoning_tokens=extracted.get("reasoning_tokens"),
        cached_input_tokens=extracted.get("cached_input_tokens"),
        extra_usage_units=extracted.get("extra_usage_units") or {},
        cache_hit=extracted.get("cache_hit"),
        started_at=started,
        completed_at=completed,
        latency_ms=int((time.monotonic() - started_monotonic) * 1000),
        **_agent_event_fields(options.get("agent")),
        attribution=options.get("attribution") or {},
        metadata=metadata,
    )


def _record_sync_httpx_event(
    client: CloptimaLLMObservability,
    event: LLMUsageEvent,
    *,
    metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]],
    fire_and_forget: bool,
) -> None:
    if fire_and_forget:
        client._record_async_with_policy(event, metadata_policy)
        return
    client._post_payload(client._event_payload(event, metadata_policy))


async def _record_async_httpx_event(
    client: CloptimaLLMObservability,
    event: LLMUsageEvent,
    *,
    metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]],
    fire_and_forget: bool,
) -> None:
    if fire_and_forget:
        client._record_async_with_policy(event, metadata_policy)
        return
    await client._apost_payload(client._event_payload(event, metadata_policy))


def _instrument_httpx_sync_call(
    client: CloptimaLLMObservability,
    *,
    call: Callable[[], Any],
    request: Any,
    options: Dict[str, Any],
    extract_usage: Optional[Callable[[Any], Dict[str, Any]]],
    on_instrumentation_error: Optional[Callable[[BaseException], None]],
) -> Any:
    started = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    try:
        response = call()
    except BaseException as exc:
        completed = datetime.now(timezone.utc)
        _record_sync_httpx_event(
            client,
            LLMUsageEvent(
                provider=options.get("provider") or "openai",
                model=options.get("model") or "unknown",
                request_id=options.get("request_id"),
                trace_id=options.get("trace_id"),
                status="failed",
                started_at=started,
                completed_at=completed,
                latency_ms=int((time.monotonic() - started_monotonic) * 1000),
                error_message=str(exc),
                **_agent_event_fields(options.get("agent")),
                attribution=options.get("attribution") or {},
                metadata=options.get("metadata") or {},
            ),
            metadata_policy=options.get("metadata_policy"),
            fire_and_forget=bool(options.get("fire_and_forget")),
        )
        raise

    completed = datetime.now(timezone.utc)
    event = _build_httpx_usage_event(
        response,
        options=options,
        extract_usage=extract_usage,
        on_instrumentation_error=on_instrumentation_error,
        started=started,
        started_monotonic=started_monotonic,
        completed=completed,
    )
    _record_sync_httpx_event(
        client,
        event,
        metadata_policy=options.get("metadata_policy"),
        fire_and_forget=bool(options.get("fire_and_forget")),
    )
    return response


async def _instrument_httpx_async_call(
    client: CloptimaLLMObservability,
    *,
    call: Callable[[], Any],
    request: Any,
    options: Dict[str, Any],
    extract_usage: Optional[Callable[[Any], Dict[str, Any]]],
    on_instrumentation_error: Optional[Callable[[BaseException], None]],
) -> Any:
    started = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    try:
        response = call()
        if hasattr(response, "__await__"):
            response = await response
    except BaseException as exc:
        completed = datetime.now(timezone.utc)
        await _record_async_httpx_event(
            client,
            LLMUsageEvent(
                provider=options.get("provider") or "openai",
                model=options.get("model") or "unknown",
                request_id=options.get("request_id"),
                trace_id=options.get("trace_id"),
                status="failed",
                started_at=started,
                completed_at=completed,
                latency_ms=int((time.monotonic() - started_monotonic) * 1000),
                error_message=str(exc),
                **_agent_event_fields(options.get("agent")),
                attribution=options.get("attribution") or {},
                metadata=options.get("metadata") or {},
            ),
            metadata_policy=options.get("metadata_policy"),
            fire_and_forget=bool(options.get("fire_and_forget")),
        )
        raise

    completed = datetime.now(timezone.utc)
    event = _build_httpx_usage_event(
        response,
        options=options,
        extract_usage=extract_usage,
        on_instrumentation_error=on_instrumentation_error,
        started=started,
        started_monotonic=started_monotonic,
        completed=completed,
    )
    await _record_async_httpx_event(
        client,
        event,
        metadata_policy=options.get("metadata_policy"),
        fire_and_forget=bool(options.get("fire_and_forget")),
    )
    return response


class _InstrumentedHttpxClientProxy:
    def __init__(
        self,
        client: Any,
        *,
        cloptima: CloptimaLLMObservability,
        provider: Optional[str],
        model: Optional[str],
        extract_usage: Optional[Callable[[Any], Dict[str, Any]]],
        attribution: Optional[Dict[str, Any]],
        agent: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
        metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]],
        request_id: Optional[str],
        trace_id: Optional[str],
        fire_and_forget: bool,
        include_headers: Optional[List[str]],
        request_id_header: Optional[str],
        trace_id_header: Optional[str],
        on_instrumentation_error: Optional[Callable[[BaseException], None]],
        resolve_options: Optional[Callable[[Any], Optional[Dict[str, Any]]]],
    ) -> None:
        self._client = client
        self._cloptima = cloptima
        self._provider = provider
        self._model = model
        self._extract_usage = extract_usage
        self._attribution = attribution
        self._agent = agent
        self._metadata = metadata
        self._metadata_policy = metadata_policy
        self._request_id = request_id
        self._trace_id = trace_id
        self._fire_and_forget = fire_and_forget
        self._include_headers = include_headers
        self._request_id_header = request_id_header
        self._trace_id_header = trace_id_header
        self._on_instrumentation_error = on_instrumentation_error
        self._resolve_options = resolve_options

    def _resolved_options(self, request: Any) -> Dict[str, Any]:
        return _resolve_httpx_instrumentation_options(
            request,
            provider=self._provider,
            model=self._model,
            attribution=self._attribution,
            agent=self._agent,
            metadata=self._metadata,
            metadata_policy=self._metadata_policy,
            request_id=self._request_id,
            trace_id=self._trace_id,
            fire_and_forget=self._fire_and_forget,
            include_headers=self._include_headers,
            request_id_header=self._request_id_header,
            trace_id_header=self._trace_id_header,
            resolve_options=self._resolve_options,
        )

    def request(self, method: Any, url: Any, *args: Any, **kwargs: Any) -> Any:
        request = _HttpxRequestProxy(method, url, kwargs.get("headers"))
        options = self._resolved_options(request)
        if not options.get("provider"):
            if self._on_instrumentation_error:
                self._on_instrumentation_error(RuntimeError("Instrumented httpx client requires a provider"))
            return self._client.request(method, url, *args, **kwargs)
        return _instrument_httpx_sync_call(
            self._cloptima,
            call=lambda: self._client.request(method, url, *args, **kwargs),
            request=request,
            options=options,
            extract_usage=self._extract_usage,
            on_instrumentation_error=self._on_instrumentation_error,
        )

    def send(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        options = self._resolved_options(request)
        if not options.get("provider"):
            if self._on_instrumentation_error:
                self._on_instrumentation_error(RuntimeError("Instrumented httpx client requires a provider"))
            return self._client.send(request, *args, **kwargs)
        return _instrument_httpx_sync_call(
            self._cloptima,
            call=lambda: self._client.send(request, *args, **kwargs),
            request=request,
            options=options,
            extract_usage=self._extract_usage,
            on_instrumentation_error=self._on_instrumentation_error,
        )

    def get(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return self.request("GET", url, *args, **kwargs)

    def post(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return self.request("POST", url, *args, **kwargs)

    def put(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return self.request("PUT", url, *args, **kwargs)

    def patch(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return self.request("PATCH", url, *args, **kwargs)

    def delete(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return self.request("DELETE", url, *args, **kwargs)

    def head(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return self.request("HEAD", url, *args, **kwargs)

    def options(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return self.request("OPTIONS", url, *args, **kwargs)

    def __enter__(self) -> Any:
        entered = self._client.__enter__() if hasattr(self._client, "__enter__") else self._client
        return self if entered is self._client else entered

    def __exit__(self, *args: Any) -> Any:
        if hasattr(self._client, "__exit__"):
            return self._client.__exit__(*args)
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _InstrumentedAsyncHttpxClientProxy(_InstrumentedHttpxClientProxy):
    async def request(self, method: Any, url: Any, *args: Any, **kwargs: Any) -> Any:
        request = _HttpxRequestProxy(method, url, kwargs.get("headers"))
        options = self._resolved_options(request)
        if not options.get("provider"):
            if self._on_instrumentation_error:
                self._on_instrumentation_error(RuntimeError("Instrumented httpx client requires a provider"))
            return await self._client.request(method, url, *args, **kwargs)
        return await _instrument_httpx_async_call(
            self._cloptima,
            call=lambda: self._client.request(method, url, *args, **kwargs),
            request=request,
            options=options,
            extract_usage=self._extract_usage,
            on_instrumentation_error=self._on_instrumentation_error,
        )

    async def send(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        options = self._resolved_options(request)
        if not options.get("provider"):
            if self._on_instrumentation_error:
                self._on_instrumentation_error(RuntimeError("Instrumented httpx client requires a provider"))
            return await self._client.send(request, *args, **kwargs)
        return await _instrument_httpx_async_call(
            self._cloptima,
            call=lambda: self._client.send(request, *args, **kwargs),
            request=request,
            options=options,
            extract_usage=self._extract_usage,
            on_instrumentation_error=self._on_instrumentation_error,
        )

    async def get(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return await self.request("GET", url, *args, **kwargs)

    async def post(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return await self.request("POST", url, *args, **kwargs)

    async def put(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return await self.request("PUT", url, *args, **kwargs)

    async def patch(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return await self.request("PATCH", url, *args, **kwargs)

    async def delete(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return await self.request("DELETE", url, *args, **kwargs)

    async def head(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return await self.request("HEAD", url, *args, **kwargs)

    async def options(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return await self.request("OPTIONS", url, *args, **kwargs)

    async def __aenter__(self) -> Any:
        entered = await self._client.__aenter__() if hasattr(self._client, "__aenter__") else self._client
        return self if entered is self._client else entered

    async def __aexit__(self, *args: Any) -> Any:
        if hasattr(self._client, "__aexit__"):
            return await self._client.__aexit__(*args)
        return False


class _InstrumentedHttpxTransportProxy(_InstrumentedHttpxClientProxy):
    def handle_request(self, request: Any) -> Any:
        options = self._resolved_options(request)
        if not options.get("provider"):
            if self._on_instrumentation_error:
                self._on_instrumentation_error(RuntimeError("Instrumented httpx transport requires a provider"))
            return self._client.handle_request(request)
        return _instrument_httpx_sync_call(
            self._cloptima,
            call=lambda: self._client.handle_request(request),
            request=request,
            options=options,
            extract_usage=self._extract_usage,
            on_instrumentation_error=self._on_instrumentation_error,
        )

    async def handle_async_request(self, request: Any) -> Any:
        options = self._resolved_options(request)
        if not options.get("provider"):
            if self._on_instrumentation_error:
                self._on_instrumentation_error(RuntimeError("Instrumented httpx transport requires a provider"))
            return await self._client.handle_async_request(request)
        return await _instrument_httpx_async_call(
            self._cloptima,
            call=lambda: self._client.handle_async_request(request),
            request=request,
            options=options,
            extract_usage=self._extract_usage,
            on_instrumentation_error=self._on_instrumentation_error,
        )


def instrument_httpx_client(
    client: Any,
    *,
    cloptima: Union[CloptimaLLMObservability, DisabledCloptimaLLMObservability],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    extract_usage: Optional[Callable[[Any], Dict[str, Any]]] = None,
    attribution: Optional[Dict[str, Any]] = None,
    agent: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    fire_and_forget: bool = True,
    include_headers: Optional[List[str]] = None,
    request_id_header: Optional[str] = None,
    trace_id_header: Optional[str] = None,
    on_instrumentation_error: Optional[Callable[[BaseException], None]] = None,
    resolve_options: Optional[Callable[[Any], Optional[Dict[str, Any]]]] = None,
) -> Any:
    if not cloptima.is_enabled():
        return client
    if hasattr(client, "send") and inspect.iscoroutinefunction(getattr(client, "send")):
        return _InstrumentedAsyncHttpxClientProxy(
            client,
            cloptima=cloptima,
            provider=provider,
            model=model,
            extract_usage=extract_usage,
            attribution=attribution,
            agent=agent,
            metadata=metadata,
            metadata_policy=metadata_policy,
            request_id=request_id,
            trace_id=trace_id,
            fire_and_forget=fire_and_forget,
            include_headers=include_headers,
            request_id_header=request_id_header,
            trace_id_header=trace_id_header,
            on_instrumentation_error=on_instrumentation_error,
            resolve_options=resolve_options,
        )
    return _InstrumentedHttpxClientProxy(
        client,
        cloptima=cloptima,
        provider=provider,
        model=model,
        extract_usage=extract_usage,
        attribution=attribution,
        agent=agent,
        metadata=metadata,
        metadata_policy=metadata_policy,
        request_id=request_id,
        trace_id=trace_id,
        fire_and_forget=fire_and_forget,
        include_headers=include_headers,
        request_id_header=request_id_header,
        trace_id_header=trace_id_header,
        on_instrumentation_error=on_instrumentation_error,
        resolve_options=resolve_options,
    )


def instrument_httpx_transport(
    transport: Any,
    *,
    cloptima: Union[CloptimaLLMObservability, DisabledCloptimaLLMObservability],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    extract_usage: Optional[Callable[[Any], Dict[str, Any]]] = None,
    attribution: Optional[Dict[str, Any]] = None,
    agent: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metadata_policy: Optional[Union[MetadataPrivacyPolicy, Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    fire_and_forget: bool = True,
    include_headers: Optional[List[str]] = None,
    request_id_header: Optional[str] = None,
    trace_id_header: Optional[str] = None,
    on_instrumentation_error: Optional[Callable[[BaseException], None]] = None,
    resolve_options: Optional[Callable[[Any], Optional[Dict[str, Any]]]] = None,
) -> Any:
    if not cloptima.is_enabled():
        return transport
    return _InstrumentedHttpxTransportProxy(
        transport,
        cloptima=cloptima,
        provider=provider,
        model=model,
        extract_usage=extract_usage,
        attribution=attribution,
        agent=agent,
        metadata=metadata,
        metadata_policy=metadata_policy,
        request_id=request_id,
        trace_id=trace_id,
        fire_and_forget=fire_and_forget,
        include_headers=include_headers,
        request_id_header=request_id_header,
        trace_id_header=trace_id_header,
        on_instrumentation_error=on_instrumentation_error,
        resolve_options=resolve_options,
    )
