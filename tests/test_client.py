from __future__ import annotations

import json
import unittest
import urllib.request
import asyncio
import builtins
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cloptima_llm_observability import (
    ainstrument_openai_compatible_response,
    ainstrument_openai_compatible_stream,
    bind_observed_async_stream,
    bind_observed_call,
    list_supported_providers,
    CloptimaLLMObservability,
    create_observed_async_stream,
    create_observed_async_call,
    create_observed_call,
    create_observed_stream,
    LLMAttribution,
    LLMUsageEvent,
    MetadataPrivacyPolicy,
    disabled_client,
    DisabledCloptimaLLMObservability,
    extract_anthropic_usage,
    extract_anthropic_stream_usage,
    extract_azure_openai_usage,
    extract_bedrock_stream_usage,
    extract_bedrock_usage,
    extract_gemini_stream_usage,
    extract_gemini_usage,
    init_from_env,
    instrument_fastapi_request_context,
    instrument_flask_request_context,
    instrument_httpx_client,
    instrument_httpx_transport,
    instrument_httpx_transport_metadata,
    instrument_openai_compatible_response,
    instrument_openai_compatible_stream,
    is_enabled,
    preview_batch_payload,
    compose_usage_extractors,
    create_mapped_usage_extractor,
    preview_event_payload,
    preview_otlp_request,
    get_provider_stream_usage_extractor,
    get_provider_usage_extractor,
    PROVIDER_SUPPORT_MATRIX,
    PROVIDER_USAGE_EXTRACTORS,
    run_with_task,
    try_extract_usage,
    extract_openai_usage,
    extract_openai_stream_usage,
    extract_vertex_stream_usage,
    extract_vertex_usage,
    run_with_attribution,
    run_with_workflow,
    task,
    validate_payload,
    with_task,
    with_usage_overrides,
    with_attribution,
    with_workflow,
    workflow,
    wrap_observed_service,
)

TEST_API_BASE_URL = "https://sdk-ingest.example.cloptima.ai"
DEFAULT_API_BASE_URL = "https://api.cloptima.ai"
TEST_INGEST_URL = f"{TEST_API_BASE_URL}/v1/ai/integrations/sdk/events"
TEST_OTLP_URL = f"{TEST_API_BASE_URL}/v1/ai/integrations/otlp/traces"
DEFAULT_INGEST_URL = f"{DEFAULT_API_BASE_URL}/v1/ai/integrations/sdk/events"
DEFAULT_OTLP_URL = f"{DEFAULT_API_BASE_URL}/v1/ai/integrations/otlp/traces"


class _FakeResponse:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b"{}"


class _FakeAsyncResponse:
    status_code = 202

    async def aread(self):
        return b"{}"


class _FakeAsyncClient:
    observed = {}
    instances = 0
    closed_count = 0

    def __init__(self, timeout):
        type(self).instances += 1
        self.timeout = timeout
        self.is_closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def post(self, url, *, content, headers):
        self.observed["url"] = url
        self.observed["content"] = content
        self.observed["headers"] = headers
        self.observed["timeout"] = self.timeout
        self.observed.setdefault("contents", []).append(content)
        return _FakeAsyncResponse()

    async def aclose(self):
        type(self).closed_count += 1
        self.is_closed = True


class _FakeHttpxResponse:
    def __init__(self, payload, *, status_code=200, headers=None, request=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.request = request

    def json(self):
        return self._payload


class _FakeHttpxClient:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls = []
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *_args):
        self.exited += 1
        return False

    def request(self, method, url, *args, **kwargs):
        self.calls.append(("request", method, url, kwargs))
        if self._error is not None:
            raise self._error
        return self._response

    def send(self, request, *args, **kwargs):
        self.calls.append(("send", request, kwargs))
        if self._error is not None:
            raise self._error
        return self._response


async def _async_chunk_stream(*chunks):
    for chunk in chunks:
        yield chunk


class _FakeAsyncHttpxClient:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls = []
        self.entered = 0
        self.exited = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *_args):
        self.exited += 1
        return False

    async def request(self, method, url, *args, **kwargs):
        self.calls.append(("request", method, url, kwargs))
        if self._error is not None:
            raise self._error
        return self._response

    async def send(self, request, *args, **kwargs):
        self.calls.append(("send", request, kwargs))
        if self._error is not None:
            raise self._error
        return self._response


class _FakeHttpxTransport:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls = []

    def handle_request(self, request):
        self.calls.append(request)
        if self._error is not None:
            raise self._error
        return self._response


class CloptimaLLMObservabilityTests(unittest.TestCase):
    def test_init_from_env_returns_silent_disabled_client_when_not_configured(self) -> None:
        client = init_from_env(env={})

        self.assertIsInstance(client, DisabledCloptimaLLMObservability)
        self.assertFalse(client.is_enabled())
        self.assertFalse(is_enabled(env={}))
        self.assertEqual(
            client.observe_call(
                provider="openai",
                model="gpt-4o-mini",
                call=lambda: "passthrough",
                feature_id="summaries",
            ),
            "passthrough",
        )

    def test_init_from_env_builds_configured_client(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["request"] = request
            return _FakeResponse()

        client = init_from_env(
            env={
                "CLOPTIMA_LLM_OBSERVABILITY_API_BASE_URL": TEST_API_BASE_URL,
                "CLOPTIMA_LLM_OBSERVABILITY_API_KEY": "pat-env",
                "CLOPTIMA_LLM_OBSERVABILITY_APP_ID": "agent-api",
                "CLOPTIMA_LLM_OBSERVABILITY_ENVIRONMENT": "prod",
                "CLOPTIMA_LLM_OBSERVABILITY_TEAM_ID": "platform",
            }
        )

        self.assertTrue(client.is_enabled())
        self.assertTrue(
            is_enabled(
                env={
                    "CLOPTIMA_LLM_OBSERVABILITY_API_BASE_URL": TEST_API_BASE_URL,
                    "CLOPTIMA_LLM_OBSERVABILITY_API_KEY": "pat-env",
                    "CLOPTIMA_LLM_OBSERVABILITY_APP_ID": "agent-api",
                    "CLOPTIMA_LLM_OBSERVABILITY_ENVIRONMENT": "prod",
                }
            )
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            client.record(LLMUsageEvent(provider="openai", model="gpt-4o-mini"))

        body = json.loads(observed["request"].data.decode("utf-8"))
        self.assertEqual(body["metadata"]["app_id"], "agent-api")
        self.assertEqual(body["metadata"]["environment"], "prod")
        self.assertEqual(body["metadata"]["team_id"], "platform")

    def test_init_from_env_uses_default_ingest_url_and_production_environment(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["request"] = request
            return _FakeResponse()

        client = init_from_env(
            env={
                "CLOPTIMA_LLM_OBSERVABILITY_API_KEY": "pat-env",
                "CLOPTIMA_LLM_OBSERVABILITY_APP_ID": "agent-api",
            }
        )

        self.assertTrue(client.is_enabled())
        self.assertTrue(
            is_enabled(
                env={
                    "CLOPTIMA_LLM_OBSERVABILITY_API_KEY": "pat-env",
                    "CLOPTIMA_LLM_OBSERVABILITY_APP_ID": "agent-api",
                }
            )
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            client.record(LLMUsageEvent(provider="openai", model="gpt-4o-mini"))

        self.assertEqual(observed["request"].full_url, DEFAULT_INGEST_URL)
        body = json.loads(observed["request"].data.decode("utf-8"))
        self.assertEqual(body["metadata"]["environment"], "production")

    def test_init_from_env_normalizes_scheme_less_and_trailing_slash_api_base_urls(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["request"] = request
            return _FakeResponse()

        client = init_from_env(
            env={
                "CLOPTIMA_LLM_OBSERVABILITY_API_KEY": "pat-env",
                "CLOPTIMA_LLM_OBSERVABILITY_APP_ID": "agent-api",
                "CLOPTIMA_LLM_OBSERVABILITY_API_BASE_URL": "sdk-ingest.example.cloptima.ai/",
            }
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            client.record(LLMUsageEvent(provider="openai", model="gpt-4o-mini"))

        self.assertEqual(observed["request"].full_url, TEST_INGEST_URL)

    def test_is_enabled_requires_api_key_and_app_id_only(self) -> None:
        self.assertFalse(
            is_enabled(
                env={
                    "CLOPTIMA_LLM_OBSERVABILITY_APP_ID": "agent-api",
                }
            )
        )
        self.assertFalse(
            is_enabled(
                env={
                    "CLOPTIMA_LLM_OBSERVABILITY_API_KEY": "pat-env",
                }
            )
        )
        self.assertTrue(
            is_enabled(
                env={
                    "CLOPTIMA_LLM_OBSERVABILITY_API_KEY": "pat-env",
                    "CLOPTIMA_LLM_OBSERVABILITY_APP_ID": "agent-api",
                }
            )
        )

    def test_direct_constructor_derives_default_otlp_url_from_default_ingest_url(self) -> None:
        observed_requests = []

        def fake_urlopen(request, timeout):
            observed_requests.append(request)
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_key="pat-env",
            default_attribution=LLMAttribution(app_id="agent-api", environment="production"),
            delivery_mode="otlp_http",
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            client.record(LLMUsageEvent(provider="openai", model="gpt-4o-mini"))

        self.assertEqual([request.full_url for request in observed_requests], [DEFAULT_OTLP_URL])

    def test_direct_constructor_inferrs_http_for_scheme_less_localhost_api_base_urls(self) -> None:
        observed_requests = []

        def fake_urlopen(request, timeout):
            observed_requests.append(request)
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url="127.0.0.1:4318/",
            api_key="pat-env",
            default_attribution=LLMAttribution(app_id="agent-api", environment="production"),
            delivery_mode="otlp_http",
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            client.record(LLMUsageEvent(provider="openai", model="gpt-4o-mini"))

        self.assertEqual(observed_requests[0].full_url, "http://127.0.0.1:4318/v1/ai/integrations/otlp/traces")
        self.assertIsNone(observed_requests[0].headers.get("Authorization"))

    def test_init_from_env_stays_fail_open_but_diagnosable_when_api_base_url_is_invalid(self) -> None:
        client = init_from_env(
            env={
                "CLOPTIMA_LLM_OBSERVABILITY_API_KEY": "pat-env",
                "CLOPTIMA_LLM_OBSERVABILITY_APP_ID": "agent-api",
                "CLOPTIMA_LLM_OBSERVABILITY_API_BASE_URL": "https://api.cloptima.ai/custom-path",
            },
            enabled=True,
        )

        self.assertFalse(client.is_enabled())
        self.assertIn("must not include a path, query, or hash", str(client.get_init_error()))

    def test_direct_constructor_rejects_dormant_dual_delivery_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, 'delivery_mode "dual" is temporarily disabled'):
            CloptimaLLMObservability(
                api_key="pat-env",
                default_attribution=LLMAttribution(app_id="agent-api", environment="production"),
                delivery_mode="dual",
            )

    def test_init_from_env_stays_fail_open_but_diagnosable_when_explicitly_enabled(self) -> None:
        observed_errors = []

        client = init_from_env(
            env={
                "CLOPTIMA_LLM_OBSERVABILITY_ENABLED": "true",
                "CLOPTIMA_LLM_OBSERVABILITY_APP_ID": "agent-api",
            },
            on_init_error=lambda error: observed_errors.append(str(error)),
        )

        self.assertFalse(client.is_enabled())
        self.assertEqual(len(observed_errors), 1)
        self.assertIn("missing required configuration", observed_errors[0])
        self.assertIn("API_KEY", observed_errors[0])
        self.assertIn("missing required configuration", str(client.get_init_error()))

    def test_disabled_client_can_be_created_directly(self) -> None:
        client = disabled_client(RuntimeError("disabled"))

        self.assertFalse(client.is_enabled())
        self.assertEqual(str(client.get_init_error()), "disabled")
        self.assertEqual(
            client.stats(),
            type(client.stats())(
                queued_events=0,
                dropped_events=0,
                delivered_events=0,
                failed_batches=0,
            ),
        )

    def test_disabled_client_observe_stream_call_passes_through_chunks(self) -> None:
        client = disabled_client()

        def source():
            yield "chunk-1"
            yield "chunk-2"

        observed = list(
            client.observe_stream_call(
                provider="openai",
                model="gpt-4o-mini",
                call=source,
            )
        )

        self.assertEqual(observed, ["chunk-1", "chunk-2"])

    def test_disabled_client_observe_async_stream_call_passes_through_chunks(self) -> None:
        client = disabled_client()

        async def source():
            yield "chunk-1"
            yield "chunk-2"

        async def collect():
            observed = []
            async for chunk in client.observe_async_stream_call(
                provider="openai",
                model="gpt-4o-mini",
                call=source,
            ):
                observed.append(chunk)
            return observed

        self.assertEqual(asyncio.run(collect()), ["chunk-1", "chunk-2"])

    def test_record_posts_canonical_payload(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["request"] = request
            observed["timeout"] = timeout
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            sdk_version="0.1.0",
            default_attribution=LLMAttribution(
                team_id="platform",
                app_id="checkout-api",
                environment="prod",
                business_unit="revenue",
                cost_center="cc-checkout",
                product="checkout",
                customer_segment="enterprise",
                end_customer_id="acct-1",
                tenant_id="tenant-1",
                release="2026.05.1",
                actor_id="svc-checkout",
                actor_type="service",
            ),
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            client.record(
                LLMUsageEvent(
                    provider="openai",
                    model="gpt-4o-mini",
                    source_event_id="event-1",
                    input_tokens=10,
                    output_tokens=5,
                    agent_session_id="agent-session-1",
                    tool_name="ticket_search",
                    retry_index=1,
                    metadata={"route": "/support"},
                )
            )

        request = observed["request"]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, TEST_INGEST_URL)
        self.assertEqual(request.headers["Authorization"], "Bearer pat-test")
        self.assertEqual(request.headers["User-agent"], "cloptima-llm-observability/0.1.0")
        self.assertEqual(body["sdk_name"], "cloptima-llm-observability")
        self.assertEqual(body["sdk_version"], "0.1.0")
        self.assertEqual(body["schema_version"], "cloptima.llm.event.v1")
        self.assertEqual(body["provider"], "openai")
        self.assertEqual(body["total_tokens"], 15)
        self.assertEqual(body["metadata"]["team_id"], "platform")
        self.assertEqual(body["metadata"]["business_unit"], "revenue")
        self.assertEqual(body["metadata"]["cost_center"], "cc-checkout")
        self.assertEqual(body["metadata"]["product"], "checkout")
        self.assertEqual(body["metadata"]["customer_segment"], "enterprise")
        self.assertEqual(body["metadata"]["end_customer_id"], "acct-1")
        self.assertEqual(body["metadata"]["tenant_id"], "tenant-1")
        self.assertEqual(body["metadata"]["release"], "2026.05.1")
        self.assertEqual(body["metadata"]["agent_session_id"], "agent-session-1")
        self.assertEqual(body["metadata"]["tool_name"], "ticket_search")
        self.assertEqual(body["metadata"]["retry_index"], 1)
        self.assertEqual(body["metadata"]["route"], "/support")
        self.assertEqual(body["sdk_delivery_stats"]["delivered_events"], 0)

    def test_record_derives_source_event_id_from_existing_identifiers_or_generates_one(self) -> None:
        observed_bodies = []

        def fake_urlopen(request, timeout):
            observed_bodies.append(json.loads(request.data.decode("utf-8")))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(
                app_id="agent-api",
                environment="dev",
            ),
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            client.record(
                LLMUsageEvent(
                    provider="openai",
                    model="gpt-4o-mini",
                    request_id="request-derive-1",
                )
            )
            client.record(
                LLMUsageEvent(
                    provider="anthropic",
                    model="claude-3-5-sonnet",
                )
            )

        self.assertEqual(observed_bodies[0]["schema_version"], "cloptima.llm.event.v1")
        self.assertEqual(observed_bodies[0]["source_event_id"], "request-derive-1")
        self.assertIsInstance(observed_bodies[1]["source_event_id"], str)
        self.assertTrue(observed_bodies[1]["source_event_id"].startswith("clop_evt_"))

    def test_record_retries_transient_sync_post_failures(self) -> None:
        attempts = {"count": 0}

        def flaky_urlopen(request, timeout):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise urllib.error.URLError("temporary unavailable")
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_retry_count=1,
            async_retry_backoff_seconds=0,
        )

        with patch.object(urllib.request, "urlopen", flaky_urlopen):
            client.record(LLMUsageEvent(provider="openai", model="gpt-4o-mini"))

        self.assertEqual(attempts["count"], 2)

    def test_observe_call_accepts_flat_attribution_fields_and_metadata_policy(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["request"] = request
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(
                app_id="agent-api",
                environment="dev",
            ),
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            result = client.observe_call(
                provider="openai",
                model="gpt-4o-mini",
                call=lambda: {
                    "id": "chatcmpl-flat-1",
                    "model": "gpt-4o-mini",
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 4,
                        "total_tokens": 13,
                    },
                },
                extract_usage=extract_openai_usage,
                team_id="platform",
                feature_id="summaries",
                workflow_id="support-agent",
                metadata={
                    "prompt": "should-be-redacted",
                    "route": "/summaries",
                },
                metadata_policy=MetadataPrivacyPolicy(
                    mode="allowlisted_metadata",
                    allowlist_keys=["route"],
                ),
                fire_and_forget=False,
            )

        self.assertEqual(result["id"], "chatcmpl-flat-1")
        body = json.loads(observed["request"].data.decode("utf-8"))
        self.assertEqual(body["metadata"]["team_id"], "platform")
        self.assertEqual(body["metadata"]["feature_id"], "summaries")
        self.assertEqual(body["metadata"]["workflow_id"], "support-agent")
        self.assertEqual(body["metadata"]["route"], "/summaries")
        self.assertNotIn("prompt", body["metadata"])

    def test_with_attribution_applies_ambient_attribution_and_preserves_explicit_overrides(self) -> None:
        observed_requests = []

        def fake_urlopen(request, timeout):
            observed_requests.append(json.loads(request.data.decode("utf-8")))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_http_client=_FakeAsyncClient(timeout=3.0),
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            with with_attribution(team_id="platform", feature_id="summaries", workflow_id="ambient-workflow"):
                client.record(LLMUsageEvent(provider="openai", model="gpt-4o-mini"))
                client.observe_call(
                    provider="openai",
                    model="gpt-4o-mini",
                    call=lambda: {
                        "id": "chatcmpl-context-1",
                        "model": "gpt-4o-mini",
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                        },
                    },
                    extract_usage=extract_openai_usage,
                    workflow_id="explicit-workflow",
                    fire_and_forget=False,
                )

        self.assertEqual(len(observed_requests), 2)
        self.assertEqual(observed_requests[0]["metadata"]["team_id"], "platform")
        self.assertEqual(observed_requests[0]["metadata"]["feature_id"], "summaries")
        self.assertEqual(observed_requests[0]["metadata"]["workflow_id"], "ambient-workflow")
        self.assertEqual(observed_requests[1]["metadata"]["team_id"], "platform")
        self.assertEqual(observed_requests[1]["metadata"]["feature_id"], "summaries")
        self.assertEqual(observed_requests[1]["metadata"]["workflow_id"], "explicit-workflow")

    def test_run_with_attribution_preserves_context_for_async_callbacks(self) -> None:
        async def callback():
            await asyncio.sleep(0)
            return preview_event_payload(
                LLMUsageEvent(provider="openai", model="gpt-4o-mini"),
                default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            )

        payload = asyncio.run(
            run_with_attribution(
                callback,
                workflow_id="async-workflow",
                feature_id="summaries",
            )
        )

        self.assertEqual(payload["metadata"]["workflow_id"], "async-workflow")
        self.assertEqual(payload["metadata"]["feature_id"], "summaries")

    def test_run_with_attribution_preserves_context_for_async_generators(self) -> None:
        async def stream():
            await asyncio.sleep(0)
            yield preview_event_payload(
                LLMUsageEvent(provider="openai", model="gpt-4o-mini"),
                default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            )["metadata"]["workflow_id"]

        async def collect() -> List[str]:
            values = []
            async for value in run_with_attribution(stream, workflow_id="async-stream"):
                values.append(value)
            return values

        self.assertEqual(asyncio.run(collect()), ["async-stream"])

    def test_run_with_attribution_rejects_non_callable_awaitables_with_helpful_error(self) -> None:
        coroutine = asyncio.sleep(0)
        try:
            with self.assertRaises(TypeError) as raised:
                run_with_attribution(coroutine, workflow_id="async-stream")
            self.assertIn("expects a zero-argument callable", str(raised.exception))
            self.assertIn("lambda: coroutine(...)", str(raised.exception))
        finally:
            coroutine.close()

    def test_workflow_and_task_helpers_set_named_attribution_defaults(self) -> None:
        observed_requests = []

        def fake_urlopen(request, timeout):
            observed_requests.append(json.loads(request.data.decode("utf-8")))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            with with_workflow("order-checkout"):
                with with_task("llm-summary"):
                    client.record(LLMUsageEvent(provider="openai", model="gpt-4o-mini"))
                with with_task("ignored-task-name"):
                    client.record(
                        LLMUsageEvent(
                            provider="openai",
                            model="gpt-4o-mini",
                            attribution={"feature_id": "explicit-feature"},
                        )
                    )

        self.assertEqual(len(observed_requests), 2)
        self.assertEqual(observed_requests[0]["metadata"]["workflow_id"], "order-checkout")
        self.assertEqual(observed_requests[0]["metadata"]["feature_id"], "llm-summary")
        self.assertEqual(observed_requests[1]["metadata"]["workflow_id"], "order-checkout")
        self.assertEqual(observed_requests[1]["metadata"]["feature_id"], "explicit-feature")

        preview = run_with_workflow(
            lambda: run_with_task(
                lambda: preview_event_payload(
                    LLMUsageEvent(provider="openai", model="gpt-4o-mini"),
                    default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
                ),
                "preview-task",
            ),
            "preview-workflow",
        )
        self.assertEqual(preview["metadata"]["workflow_id"], "preview-workflow")
        self.assertEqual(preview["metadata"]["feature_id"], "preview-task")

    def test_client_instance_context_manager_helpers_delegate_to_module_context(self) -> None:
        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with client.with_workflow("instance-workflow", team_id="platform"):
            with client.with_task("instance-task", tenant_id="acme-prod"):
                preview = preview_event_payload(
                    LLMUsageEvent(provider="openai", model="gpt-4o-mini"),
                    default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
                )

        self.assertEqual(preview["metadata"]["workflow_id"], "instance-workflow")
        self.assertEqual(preview["metadata"]["feature_id"], "instance-task")
        self.assertEqual(preview["metadata"]["team_id"], "platform")
        self.assertEqual(preview["metadata"]["tenant_id"], "acme-prod")

    def test_workflow_and_task_decorators_apply_context_for_sync_and_async_functions(self) -> None:
        @workflow("billing-run")
        @task("summary-step")
        def sync_payload():
            return preview_event_payload(
                LLMUsageEvent(provider="openai", model="gpt-4o-mini"),
                default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            )

        @workflow("async-billing-run")
        @task("async-summary-step")
        async def async_payload():
            await asyncio.sleep(0)
            return preview_event_payload(
                LLMUsageEvent(provider="openai", model="gpt-4o-mini"),
                default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            )

        sync_result = sync_payload()
        async_result = asyncio.run(async_payload())

        self.assertEqual(sync_result["metadata"]["workflow_id"], "billing-run")
        self.assertEqual(sync_result["metadata"]["feature_id"], "summary-step")
        self.assertEqual(async_result["metadata"]["workflow_id"], "async-billing-run")
        self.assertEqual(async_result["metadata"]["feature_id"], "async-summary-step")

    def test_create_observed_call_and_async_stream_reduce_wrapper_boilerplate(self) -> None:
        observed_requests = []
        _FakeAsyncClient.observed = {}

        def fake_urlopen(request, timeout):
            observed_requests.append(json.loads(request.data.decode("utf-8")))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_http_client=_FakeAsyncClient(timeout=3.0),
        )

        observe_openai = create_observed_call(
            client,
            provider="openai",
            model="gpt-4o-mini",
            extract_usage=extract_openai_usage,
            attribution={"feature_id": "wrapper-default"},
            metadata={"channel": "sync"},
            fire_and_forget=False,
        )

        observe_stream = create_observed_async_stream(
            client,
            provider="anthropic",
            model="claude-3-5-sonnet",
            extract_usage=extract_anthropic_stream_usage,
            metadata={"channel": "stream"},
            fire_and_forget=False,
        )

        async def run_stream():
            emitted = []
            async for chunk in observe_stream(
                lambda: _async_chunk_stream(
                    {"message": {"id": "msg-factory-1", "model": "claude-3-5-sonnet"}},
                    {"usage": {"input_tokens": 4, "output_tokens": 2}},
                ),
                attribution={"workflow_id": "wrapper-stream"},
            ):
                emitted.append(chunk)
            return emitted

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            response = observe_openai(
                lambda: {
                    "id": "chatcmpl-factory-1",
                    "model": "gpt-4o-mini",
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                },
                attribution={"workflow_id": "wrapper-invoke"},
                metadata={"operation": "summarize"},
            )
            emitted = asyncio.run(run_stream())

        self.assertEqual(response["id"], "chatcmpl-factory-1")
        self.assertEqual(len(emitted), 2)
        self.assertEqual(len(observed_requests), 1)
        self.assertEqual(observed_requests[0]["metadata"]["feature_id"], "wrapper-default")
        self.assertEqual(observed_requests[0]["metadata"]["workflow_id"], "wrapper-invoke")
        self.assertEqual(observed_requests[0]["metadata"]["channel"], "sync")
        self.assertEqual(observed_requests[0]["metadata"]["operation"], "summarize")
        async_body = json.loads(_FakeAsyncClient.observed["content"].decode("utf-8"))
        self.assertEqual(async_body["metadata"]["workflow_id"], "wrapper-stream")
        self.assertEqual(async_body["metadata"]["channel"], "stream")

    def test_create_observed_async_call_and_sync_stream_cover_remaining_wrapper_helpers(self) -> None:
        observed_requests = []
        _FakeAsyncClient.observed = {}

        def fake_urlopen(request, timeout):
            observed_requests.append(json.loads(request.data.decode("utf-8")))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_http_client=_FakeAsyncClient(timeout=3.0),
        )

        observe_async = create_observed_async_call(
            client,
            provider="openai",
            model="gpt-4o-mini",
            extract_usage=extract_openai_usage,
            metadata={"channel": "async-call"},
            fire_and_forget=False,
        )
        observe_sync_stream = create_observed_stream(
            client,
            provider="anthropic",
            model="claude-3-5-sonnet",
            extract_usage=extract_anthropic_stream_usage,
            metadata={"channel": "sync-stream"},
            fire_and_forget=False,
        )

        async def run_async_call():
            return await observe_async(
                lambda: {
                    "id": "chatcmpl-factory-async-1",
                    "model": "gpt-4o-mini",
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                },
                attribution={"workflow_id": "wrapper-async-call"},
            )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            response = asyncio.run(run_async_call())
            emitted = list(
                observe_sync_stream(
                    lambda: iter([
                        {"message": {"id": "msg-factory-sync-1", "model": "claude-3-5-sonnet"}},
                        {"usage": {"input_tokens": 7, "output_tokens": 2}},
                    ]),
                    attribution={"workflow_id": "wrapper-sync-stream"},
                )
            )

        self.assertEqual(response["id"], "chatcmpl-factory-async-1")
        self.assertEqual(len(emitted), 2)
        async_body = json.loads(_FakeAsyncClient.observed["content"].decode("utf-8"))
        self.assertEqual(async_body["metadata"]["workflow_id"], "wrapper-async-call")
        self.assertEqual(async_body["metadata"]["channel"], "async-call")
        self.assertEqual(observed_requests[0]["metadata"]["workflow_id"], "wrapper-sync-stream")
        self.assertEqual(observed_requests[0]["metadata"]["channel"], "sync-stream")

    def test_bind_observed_call_and_async_stream_wrap_existing_service_methods(self) -> None:
        observed_requests = []
        _FakeAsyncClient.observed = {}

        def fake_urlopen(request, timeout):
            observed_requests.append(json.loads(request.data.decode("utf-8")))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_http_client=_FakeAsyncClient(timeout=3.0),
        )

        class ProviderService:
            def generate(self, prompt, request_id):
                return {
                    "id": f"chatcmpl-{request_id}",
                    "model": "gpt-4o-mini",
                    "usage": {
                        "prompt_tokens": len(prompt),
                        "completion_tokens": 2,
                        "total_tokens": len(prompt) + 2,
                    },
                }

            async def stream(self, prompt, request_id):
                async for chunk in _async_chunk_stream(
                    {"message": {"id": f"msg-{request_id}", "model": "claude-3-5-sonnet"}},
                    {"usage": {"input_tokens": len(prompt), "output_tokens": 1}},
                ):
                    yield chunk

        service = ProviderService()
        observed_generate = bind_observed_call(
            client,
            service.generate,
            provider="openai",
            model="gpt-4o-mini",
            extract_usage=extract_openai_usage,
            metadata={"service": "provider-service"},
            fire_and_forget=False,
            resolve_overrides=lambda _prompt, request_id: {
                "request_id": request_id,
                "attribution": {"workflow_id": f"wf-{request_id}"},
            },
        )
        observed_stream = bind_observed_async_stream(
            client,
            service.stream,
            provider="anthropic",
            model="claude-3-5-sonnet",
            extract_usage=extract_anthropic_stream_usage,
            metadata={"service": "provider-service-stream"},
            fire_and_forget=False,
            resolve_overrides=lambda _prompt, request_id: {
                "request_id": request_id,
                "attribution": {"workflow_id": f"wf-{request_id}"},
            },
        )

        async def run_stream():
            emitted = []
            async for chunk in observed_stream("hey", "req-456"):
                emitted.append(chunk)
            return emitted

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            response = observed_generate("hello", "req-123")
            emitted = asyncio.run(run_stream())

        self.assertEqual(response["id"], "chatcmpl-req-123")
        self.assertEqual(len(emitted), 2)
        self.assertEqual(observed_requests[0]["request_id"], "req-123")
        self.assertEqual(observed_requests[0]["metadata"]["workflow_id"], "wf-req-123")
        self.assertEqual(observed_requests[0]["metadata"]["service"], "provider-service")
        async_body = json.loads(_FakeAsyncClient.observed["content"].decode("utf-8"))
        self.assertEqual(async_body["request_id"], "req-456")
        self.assertEqual(async_body["metadata"]["workflow_id"], "wf-req-456")
        self.assertEqual(async_body["metadata"]["service"], "provider-service-stream")

    def test_record_posts_otlp_json_when_otlp_delivery_mode_is_enabled(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["request"] = request
            observed["timeout"] = timeout
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            sdk_version="0.1.0",
            delivery_mode="otlp_http",
            otlp_service_name="checkout-api",
            otlp_service_version="2026.06.1",
            default_attribution=LLMAttribution(
                team_id="platform",
                app_id="checkout-api",
                feature_id="support-agent",
                environment="prod",
            ),
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            client.record(
                LLMUsageEvent(
                    provider="openai",
                    model="gpt-4o-mini",
                    source_event_id="event-otlp-1",
                    request_id="request-otlp-1",
                    provider_request_id="chatcmpl-otlp-1",
                    input_tokens=10,
                    output_tokens=5,
                    vendor_reported_cost_usd="0.0123",
                    cache_hit=True,
                )
            )

        request = observed["request"]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, TEST_OTLP_URL)
        self.assertEqual(request.headers["Authorization"], "Bearer pat-test")
        self.assertEqual(request.headers["User-agent"], "cloptima-llm-observability/0.1.0")
        self.assertEqual(body["resourceSpans"][0]["resource"]["attributes"][0]["value"]["stringValue"], "checkout-api")
        self.assertEqual(body["resourceSpans"][0]["resource"]["attributes"][1]["value"]["stringValue"], "2026.06.1")
        self.assertEqual(body["resourceSpans"][0]["scopeSpans"][0]["scope"]["name"], "cloptima-llm-observability")
        self.assertEqual(body["resourceSpans"][0]["scopeSpans"][0]["scope"]["version"], "0.1.0")
        span = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        self.assertEqual(span["name"], "llm.openai.gpt-4o-mini")
        attrs = {attribute["key"]: attribute["value"] for attribute in span["attributes"]}
        self.assertEqual(attrs["gen_ai.system"]["stringValue"], "openai")
        self.assertEqual(attrs["gen_ai.request.model"]["stringValue"], "gpt-4o-mini")
        self.assertEqual(attrs["gen_ai.request.id"]["stringValue"], "request-otlp-1")
        self.assertEqual(attrs["gen_ai.response.id"]["stringValue"], "chatcmpl-otlp-1")
        self.assertEqual(attrs["source_event_id"]["stringValue"], "event-otlp-1")
        self.assertEqual(attrs["gen_ai.usage.input_tokens"]["intValue"], 10)
        self.assertEqual(attrs["gen_ai.usage.output_tokens"]["intValue"], 5)
        self.assertEqual(attrs["gen_ai.usage.total_tokens"]["intValue"], 15)
        self.assertEqual(attrs["gen_ai.usage.cost"]["doubleValue"], 0.0123)
        self.assertEqual(attrs["cache_hit"]["boolValue"], True)
        self.assertEqual(attrs["team_id"]["stringValue"], "platform")
        self.assertEqual(attrs["app_id"]["stringValue"], "checkout-api")
        self.assertEqual(attrs["feature_id"]["stringValue"], "support-agent")
        self.assertEqual(attrs["environment"]["stringValue"], "prod")

    def test_record_does_not_leak_cloptima_authorization_to_non_cloptima_otlp_hosts(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["request"] = request
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url="http://127.0.0.1:4318",
            api_key="pat-test",
            delivery_mode="otlp_http",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            client.record(LLMUsageEvent(provider="openai", model="gpt-4o-mini", source_event_id="event-otlp-local-1"))

        self.assertEqual(observed["request"].full_url, "http://127.0.0.1:4318/v1/ai/integrations/otlp/traces")
        self.assertIsNone(observed["request"].headers.get("Authorization"))

    def test_record_applies_metadata_privacy_rules_before_ingest(self) -> None:
        observed = {}
        drops = []

        def fake_urlopen(request, timeout):
            observed["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            metadata_policy=MetadataPrivacyPolicy(
                mode="metadata_only",
                hash_keys=["session_id"],
                max_value_length=8,
                on_metadata_drop=drops.append,
            ),
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            client.record(
                LLMUsageEvent(
                    provider="openai",
                    model="gpt-4o-mini",
                    metadata={
                        "prompt": "top secret prompt text",
                        "session_id": "session-123",
                        "note": "abcdefghijklmno",
                        "nested": {"path": "/chat"},
                    },
                )
            )

        body = observed["body"]
        self.assertEqual(body["metadata"]["prompt"], "[redacted]")
        self.assertTrue(body["metadata"]["session_id"].startswith("sha256_"))
        self.assertEqual(body["metadata"]["note"], "abcdefgh…")
        self.assertEqual(body["metadata"]["nested"]["path"], "/chat")
        self.assertEqual(body["metadata"]["app_id"], "agent-api")
        self.assertEqual(body["metadata"]["environment"], "dev")
        self.assertTrue(any(entry["reason"] == "redacted" and entry["key_path"] == "prompt" for entry in drops))
        self.assertTrue(any(entry["reason"] == "hashed" and entry["key_path"] == "session_id" for entry in drops))

    def test_record_strict_finops_mode_keeps_only_finance_safe_custom_metadata_keys(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            metadata_policy=MetadataPrivacyPolicy(mode="strict_finops"),
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            client.record(
                LLMUsageEvent(
                    provider="openai",
                    model="gpt-4o-mini",
                    metadata={
                        "route": "/chat",
                        "conversation_id": "conv-1",
                        "prompt": "should-not-leak",
                    },
                )
            )

        body = observed["body"]
        self.assertEqual(body["metadata"]["route"], "/chat")
        self.assertNotIn("conversation_id", body["metadata"])
        self.assertNotIn("prompt", body["metadata"])

    def test_init_from_env_stays_fail_open_but_diagnosable_when_dual_delivery_mode_is_requested(self) -> None:
        client = init_from_env(
            env={
                "CLOPTIMA_LLM_OBSERVABILITY_API_KEY": "pat-env",
                "CLOPTIMA_LLM_OBSERVABILITY_APP_ID": "agent-api",
                "CLOPTIMA_LLM_OBSERVABILITY_DELIVERY_MODE": "dual",
            }
        )

        self.assertFalse(client.is_enabled())
        self.assertIn('delivery_mode "dual" is temporarily disabled', str(client.get_init_error()))

    def test_observe_records_successful_openai_usage(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            response = client.observe(
                provider="openai",
                model="gpt-4o-mini",
                fire_and_forget=False,
                agent={
                    "agent_session_id": "agent-session-2",
                    "tool_name": "profile_lookup",
                    "retry_index": "1",
                    "loop_iteration": "2",
                },
                metadata={
                    "retry_index": "bad",
                    "loop_iteration": "bad",
                },
                call=lambda: {
                    "id": "chatcmpl-1",
                    "model": "gpt-4o-mini",
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                        "total_tokens": 10,
                        "prompt_tokens_details": {
                            "cached_tokens": 2,
                            "cache_creation_input_tokens_5m": 4,
                        },
                        "completion_tokens_details": {
                            "reasoning_tokens": 1,
                        },
                    },
                },
                extract_usage=extract_openai_usage,
            )

        self.assertEqual(response["id"], "chatcmpl-1")
        self.assertEqual(observed["body"]["provider_request_id"], "chatcmpl-1")
        self.assertEqual(observed["body"]["metadata"]["agent_session_id"], "agent-session-2")
        self.assertEqual(observed["body"]["metadata"]["tool_name"], "profile_lookup")
        self.assertEqual(observed["body"]["metadata"]["retry_index"], 1)
        self.assertEqual(observed["body"]["metadata"]["loop_iteration"], 2)
        self.assertEqual(observed["body"]["input_tokens"], 7)
        self.assertEqual(observed["body"]["output_tokens"], 3)
        self.assertEqual(observed["body"]["reasoning_tokens"], 1)
        self.assertEqual(observed["body"]["cached_input_tokens"], 2)
        self.assertEqual(observed["body"]["extra_usage_units"], {"cache_write_5m": 4})

    def test_observe_preserves_vendor_reported_cost_from_custom_extractor(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        extractor = create_mapped_usage_extractor(
            defaults={"provider": "vertex_ai"},
            fields={
                "provider_request_id": "response.id",
                "model": "response.model",
                "vendor_reported_cost_usd": "billing.cost_usd",
            },
            number_fields={
                "input_tokens": "usage.prompt_tokens",
                "output_tokens": "usage.completion_tokens",
            },
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            client.observe(
                provider="vertex_ai",
                model="gemini-2.5-pro",
                fire_and_forget=False,
                call=lambda: {
                    "response": {"id": "resp-cost-1", "model": "gemini-2.5-pro"},
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8},
                    "billing": {"cost_usd": "0.4321"},
                },
                extract_usage=extractor,
            )

        self.assertEqual(observed["body"]["provider_request_id"], "resp-cost-1")
        self.assertEqual(observed["body"]["vendor_reported_cost_usd"], 0.4321)

    def test_instrument_openai_compatible_response_records_existing_response(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        existing_response = {
            "id": "chatcmpl-helper-1",
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 6,
                "total_tokens": 10,
            },
        }

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            response = instrument_openai_compatible_response(
                client,
                existing_response,
                fire_and_forget=False,
                metadata={"integration_mode": "passive_helper"},
            )

        self.assertEqual(response, existing_response)
        self.assertEqual(observed["body"]["provider"], "openai")
        self.assertEqual(observed["body"]["provider_request_id"], "chatcmpl-helper-1")
        self.assertEqual(observed["body"]["input_tokens"], 4)
        self.assertEqual(observed["body"]["output_tokens"], 6)
        self.assertEqual(observed["body"]["metadata"]["integration_mode"], "passive_helper")
        self.assertIsNone(observed["body"].get("latency_ms"))

    def test_instrument_openai_compatible_response_measures_real_latency_when_wrapping_the_provider_call(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        def provider_call():
            time.sleep(0.015)
            return {
                "id": "chatcmpl-helper-latency-1",
                "model": "gpt-4o-mini",
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 6,
                    "total_tokens": 10,
                },
            }

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            response = instrument_openai_compatible_response(
                client,
                provider_call,
                fire_and_forget=False,
            )

        self.assertEqual(response["id"], "chatcmpl-helper-latency-1")
        self.assertGreaterEqual(observed["body"]["latency_ms"], 10)

    def test_provider_usage_extractors_normalize_common_response_shapes(self) -> None:
        self.assertEqual(
            extract_anthropic_usage({
                "id": "msg-1",
                "model": "claude-3-5-sonnet",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 3,
                    "cache_creation_input_tokens": 7,
                    "input_audio_tokens": 4,
                },
            }),
            {
                "provider": "anthropic",
                "provider_request_id": "msg-1",
                "model": "claude-3-5-sonnet",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cached_input_tokens": 3,
                "extra_usage_units": {"cache_write": 7, "input_audio": 4},
                "cache_hit": True,
            },
        )
        self.assertEqual(
            extract_gemini_usage({
                "responseId": "gemini-response-1",
                "modelVersion": "gemini-2.5-flash",
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 13,
                    "totalTokenCount": 24,
                    "thoughtsTokenCount": 2,
                    "cachedContentTokenCount": 4,
                    "promptTokensDetails": {
                        "audioTokenCount": 5,
                        "imageTokenCount": 3,
                    },
                    "candidatesTokensDetails": {
                        "videoTokenCount": 2,
                    },
                },
            }),
            {
                "provider": "gemini",
                "provider_request_id": "gemini-response-1",
                "model": "gemini-2.5-flash",
                "input_tokens": 11,
                "output_tokens": 13,
                "total_tokens": 24,
                "reasoning_tokens": 2,
                "cached_input_tokens": 4,
                "extra_usage_units": {
                    "input_audio": 5,
                    "input_image": 3,
                    "output_video": 2,
                },
                "cache_hit": True,
            },
        )
        self.assertEqual(
            extract_gemini_usage({
                "responseId": "gemini-response-list-1",
                "modelVersion": "gemini-2.5-flash",
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 13,
                    "totalTokenCount": 24,
                    "promptTokensDetails": [
                        {"modality": "AUDIO", "tokenCount": 5},
                        {"modality": "IMAGE", "tokenCount": 3},
                    ],
                    "candidatesTokensDetails": [
                        {"modality": "VIDEO", "tokenCount": 2},
                    ],
                },
            }),
            {
                "provider": "gemini",
                "provider_request_id": "gemini-response-list-1",
                "model": "gemini-2.5-flash",
                "input_tokens": 11,
                "output_tokens": 13,
                "total_tokens": 24,
                "extra_usage_units": {
                    "input_audio": 5,
                    "input_image": 3,
                    "output_video": 2,
                },
            },
        )

        self.assertEqual(
            extract_vertex_usage({
                "response_id": "vertex-response-1",
                "model_version": "gemini-2.5-pro",
                "usage_metadata": {
                    "prompt_token_count": 3,
                    "candidates_token_count": 4,
                    "total_token_count": 7,
                },
            })["provider"],
            "vertex_ai",
        )
        self.assertEqual(
            extract_bedrock_usage({
                "modelId": "anthropic.claude-3-5-sonnet",
                "usage": {
                    "inputTokens": 20,
                    "outputTokens": 6,
                    "totalTokens": 26,
                    "inputAudioTokens": 7,
                    "completionTokensDetails": {"imageTokenCount": 2},
                },
                "metrics": {"latencyMs": 321},
                "ResponseMetadata": {"RequestId": "bedrock-request-1"},
            }),
            {
                "provider": "bedrock",
                "provider_request_id": "bedrock-request-1",
                "model": "anthropic.claude-3-5-sonnet",
                "input_tokens": 20,
                "output_tokens": 6,
                "total_tokens": 26,
                "extra_usage_units": {"input_audio": 7, "output_image": 2},
                "latency_ms": 321,
            },
        )

    def test_stream_usage_aggregators_tolerate_cumulative_counters(self) -> None:
        self.assertEqual(
            extract_anthropic_stream_usage([
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg-stream-cumulative",
                        "model": "claude-3-5-sonnet",
                        "usage": {"input_tokens": 8, "input_audio_tokens": 2},
                    },
                },
                {
                    "type": "message_delta",
                    "usage": {"output_tokens": 2, "cache_creation_input_tokens": 1, "input_audio_tokens": 2},
                },
                {
                    "type": "message_delta",
                    "usage": {"output_tokens": 4, "cache_creation_input_tokens": 3, "input_audio_tokens": 5},
                },
            ]),
            {
                "provider": "anthropic",
                "provider_request_id": "msg-stream-cumulative",
                "model": "claude-3-5-sonnet",
                "input_tokens": 8,
                "output_tokens": 4,
                "total_tokens": 12,
                "extra_usage_units": {"cache_write": 3, "input_audio": 5},
            },
        )
        self.assertEqual(
            extract_bedrock_stream_usage([
                {
                    "requestId": "bedrock-stream-cumulative",
                    "modelId": "anthropic.claude-3-5-sonnet",
                    "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6, "outputVideoTokens": 1},
                },
                {
                    "requestId": "bedrock-stream-cumulative",
                    "modelId": "anthropic.claude-3-5-sonnet",
                    "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8, "outputVideoTokens": 2},
                },
            ]),
            {
                "provider": "bedrock",
                "provider_request_id": "bedrock-stream-cumulative",
                "model": "anthropic.claude-3-5-sonnet",
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8,
                "extra_usage_units": {"output_video": 2},
            },
        )
        self.assertEqual(
            extract_azure_openai_usage({
                "id": "chatcmpl-azure",
                "deployment_name": "gpt-4o-mini-prod",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                    "completion_tokens_details": {"image_tokens": 4},
                },
            }),
            {
                "provider": "azure_openai",
                "provider_request_id": "chatcmpl-azure",
                "model": "gpt-4o-mini-prod",
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "extra_usage_units": {"output_image": 4},
            },
        )

    def test_provider_extractors_accept_object_like_responses_and_composition_helpers(self) -> None:
        class OpenAIModel:
            def model_dump(self):
                return {
                    "id": "chatcmpl-model-dump",
                    "model": "gpt-4o-mini",
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 5,
                        "total_tokens": 13,
                    },
                }

        class GeminiModel:
            def dict(self):
                return {
                    "responseId": "gemini-dict-1",
                    "modelVersion": "gemini-2.5-flash",
                    "usageMetadata": {
                        "promptTokenCount": 6,
                        "responseTokenCount": 4,
                        "totalTokenCount": 10,
                    },
                }

        class BedrockModel:
            def toJSON(self):
                return {
                    "request_id": "bedrock-json-1",
                    "model_id": "anthropic.claude-3-5-sonnet",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "total_tokens": 15,
                    },
                }

        self.assertEqual(
            extract_openai_usage(OpenAIModel()),
            {
                "provider": "openai",
                "provider_request_id": "chatcmpl-model-dump",
                "model": "gpt-4o-mini",
                "input_tokens": 8,
                "output_tokens": 5,
                "total_tokens": 13,
            },
        )
        self.assertEqual(
            extract_gemini_usage(GeminiModel()),
            {
                "provider": "gemini",
                "provider_request_id": "gemini-dict-1",
                "model": "gemini-2.5-flash",
                "input_tokens": 6,
                "output_tokens": 4,
                "total_tokens": 10,
            },
        )
        self.assertEqual(
            extract_bedrock_usage(BedrockModel()),
            {
                "provider": "bedrock",
                "provider_request_id": "bedrock-json-1",
                "model": "anthropic.claude-3-5-sonnet",
                "input_tokens": 12,
                "output_tokens": 3,
                "total_tokens": 15,
            },
        )

        fallback_extractor = compose_usage_extractors(
            lambda _: {},
            extract_openai_usage,
        )
        self.assertEqual(
            fallback_extractor(OpenAIModel()),
            {
                "provider": "openai",
                "provider_request_id": "chatcmpl-model-dump",
                "model": "gpt-4o-mini",
                "input_tokens": 8,
                "output_tokens": 5,
                "total_tokens": 13,
            },
        )

        self.assertEqual(
            try_extract_usage(GeminiModel(), lambda _: {}, extract_gemini_usage),
            {
                "provider": "gemini",
                "provider_request_id": "gemini-dict-1",
                "model": "gemini-2.5-flash",
                "input_tokens": 6,
                "output_tokens": 4,
                "total_tokens": 10,
            },
        )

        overridden = with_usage_overrides(
            extract_anthropic_usage,
            lambda extracted, _input: {**extracted, "output_tokens": 9},
        )
        self.assertEqual(
            overridden(
                {
                    "id": "msg-override-1",
                    "model": "claude-3-5-sonnet",
                    "usage": {
                        "input_tokens": 4,
                        "output_tokens": 2,
                        "total_tokens": 6,
                    },
                }
            ),
            {
                "provider": "anthropic",
                "provider_request_id": "msg-override-1",
                "model": "claude-3-5-sonnet",
                "input_tokens": 4,
                "output_tokens": 9,
                "total_tokens": 6,
            },
        )

    def test_create_mapped_usage_extractor_maps_nested_custom_payloads(self) -> None:
        extractor = create_mapped_usage_extractor(
            defaults={
                "provider": "custom_provider",
            },
            fields={
                "provider_request_id": ["response.id", "meta.request_id"],
                "model": "meta.model_name",
                "status": "meta.status",
            },
            number_fields={
                "input_tokens": "usage.input",
                "output_tokens": "usage.output",
                "total_tokens": "usage.total",
                "latency_ms": "timing.latency_ms",
            },
            boolean_fields={
                "cache_hit": "cache.hit",
            },
            extra_usage_units={
                "images": "usage.images_generated",
            },
            metadata={
                "region": "meta.region",
                "route": "meta.route",
            },
        )

        self.assertEqual(
            extractor(
                {
                    "response": {"id": "resp-custom-1"},
                    "meta": {
                        "model_name": "custom-model",
                        "status": "succeeded",
                        "region": "us-central1",
                        "route": "/v1/generate",
                    },
                    "usage": {
                        "input": 7,
                        "output": 3,
                        "total": 10,
                        "images_generated": 2,
                    },
                    "timing": {"latency_ms": 145},
                    "cache": {"hit": True},
                }
            ),
            {
                "provider": "custom_provider",
                "provider_request_id": "resp-custom-1",
                "model": "custom-model",
                "status": "succeeded",
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
                "latency_ms": 145,
                "cache_hit": True,
                "extra_usage_units": {"images": 2},
                "metadata": {
                    "region": "us-central1",
                    "route": "/v1/generate",
                },
            },
        )

    def test_provider_extractor_registry_resolves_aliases_and_fixture_coverage_stays_aligned(self) -> None:
        self.assertIs(get_provider_usage_extractor("azure"), extract_azure_openai_usage)
        self.assertIs(get_provider_usage_extractor("vertex-ai"), extract_vertex_usage)
        self.assertIs(get_provider_stream_usage_extractor("bedrock"), extract_bedrock_stream_usage)
        self.assertIsNone(get_provider_usage_extractor(None))
        self.assertIsNone(get_provider_stream_usage_extractor(None))
        self.assertTrue(any(descriptor["provider"] == "openai" for descriptor in PROVIDER_USAGE_EXTRACTORS))
        self.assertEqual(
            list_supported_providers(),
            [{**dict(descriptor), "aliases": list(descriptor["aliases"])} for descriptor in PROVIDER_SUPPORT_MATRIX],
        )
        self.assertTrue(all(descriptor["response"] is True for descriptor in PROVIDER_SUPPORT_MATRIX))
        self.assertTrue(any(descriptor["provider"] == "anthropic" and descriptor["stream"] is True for descriptor in PROVIDER_SUPPORT_MATRIX))
        with self.assertRaises(AttributeError):
            PROVIDER_USAGE_EXTRACTORS.append({})  # type: ignore[attr-defined]
        with self.assertRaises(TypeError):
            PROVIDER_USAGE_EXTRACTORS[0]["provider"] = "mutated"  # type: ignore[index]
        with self.assertRaises(TypeError):
            PROVIDER_USAGE_EXTRACTORS[0]["aliases"] += ("mutated",)  # type: ignore[index]

        fixture_candidates = [
            Path(__file__).resolve().parents[2] / "llm-observability-fixtures" / "provider_usage_replay.json",
            Path(__file__).resolve().parents[1] / "llm-observability-fixtures" / "provider_usage_replay.json",
        ]
        fixture_path = next((path for path in fixture_candidates if path.exists()), None)
        if fixture_path is None:
            raise FileNotFoundError("provider_usage_replay.json fixture not found")
        fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
        for fixture in fixtures:
            extractor = (
                get_provider_stream_usage_extractor(fixture["provider"])
                if fixture["kind"] == "stream"
                else get_provider_usage_extractor(fixture["provider"])
            )
            self.assertIsNotNone(extractor, f"missing registry extractor for {fixture['provider']}/{fixture['kind']}")
            self.assertEqual(
                extractor(fixture["payload"]),
                fixture["expected"],
                fixture["name"],
            )

    def test_wrap_observed_service_wraps_multiple_existing_service_methods_together(self) -> None:
        observed_requests = []
        _FakeAsyncClient.observed = {}

        def fake_urlopen(request, timeout):
            observed_requests.append(json.loads(request.data.decode("utf-8")))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_http_client=_FakeAsyncClient(timeout=3.0),
        )

        class SharedAIService:
            def plain_helper(self):
                return "helper-ok"

            def summarize(self, text, request_id):
                return {
                    "id": f"chatcmpl-{request_id}",
                    "model": "gpt-4o-mini",
                    "usage": {
                        "prompt_tokens": len(text),
                        "completion_tokens": 2,
                        "total_tokens": len(text) + 2,
                    },
                }

            async def stream_reply(self, text, request_id):
                async for chunk in _async_chunk_stream(
                    {"message": {"id": f"msg-{request_id}", "model": "claude-3-5-sonnet"}},
                    {"usage": {"input_tokens": len(text), "output_tokens": 1}},
                ):
                    yield chunk

        wrapped = wrap_observed_service(
            client,
            SharedAIService(),
            {
                "summarize": {
                    "kind": "call",
                    "options": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "extract_usage": extract_openai_usage,
                        "fire_and_forget": False,
                    },
                    "resolve_overrides": lambda _text, request_id: {"request_id": request_id},
                },
                "stream_reply": {
                    "kind": "async_stream",
                    "options": {
                        "provider": "anthropic",
                        "model": "claude-3-5-sonnet",
                        "extract_usage": extract_anthropic_stream_usage,
                        "fire_and_forget": False,
                    },
                    "resolve_overrides": lambda _text, request_id: {"request_id": request_id},
                },
            },
        )

        async def run_stream():
            emitted = []
            async for chunk in wrapped.stream_reply("hey", "svc-2"):
                emitted.append(chunk)
            return emitted

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            response = wrapped.summarize("hello", "svc-1")
            emitted = asyncio.run(run_stream())

        self.assertEqual(response["id"], "chatcmpl-svc-1")
        self.assertEqual(len(emitted), 2)
        self.assertEqual(wrapped.plain_helper(), "helper-ok")
        self.assertEqual(observed_requests[0]["request_id"], "svc-1")
        async_body = json.loads(_FakeAsyncClient.observed["content"].decode("utf-8"))
        self.assertEqual(async_body["request_id"], "svc-2")

    def test_observe_records_failure_when_synchronous(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                client.observe(
                    provider="anthropic",
                    model="claude-3-5-sonnet",
                    fire_and_forget=False,
                    call=lambda: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
                )

        self.assertEqual(observed["body"]["status"], "failed")
        self.assertEqual(observed["body"]["provider"], "anthropic")
        self.assertEqual(observed["body"]["error_message"], "provider unavailable")

    def test_arecord_posts_canonical_payload_without_threading(self) -> None:
        observed = {}

        class FakeAsyncClient(_FakeAsyncClient):
            instances = 0
            closed_count = 0

        FakeAsyncClient.observed = observed

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            timeout_seconds=3,
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with patch.dict(sys.modules, {"httpx": SimpleNamespace(AsyncClient=FakeAsyncClient)}):
            asyncio.run(
                client.arecord(
                    LLMUsageEvent(
                        provider="openai",
                        model="gpt-4o-mini",
                        source_event_id="event-async",
                        input_tokens=2,
                        output_tokens=4,
                    )
                )
            )

        body = json.loads(observed["content"].decode("utf-8"))
        self.assertEqual(observed["url"], TEST_INGEST_URL)
        self.assertEqual(observed["headers"]["authorization"], "Bearer pat-test")
        self.assertEqual(observed["headers"]["user-agent"], "cloptima-llm-observability/0.2.0")
        self.assertEqual(observed["timeout"], 3)
        self.assertEqual(body["provider"], "openai")
        self.assertEqual(body["total_tokens"], 6)

    def test_arecord_reuses_persistent_async_client_until_aclose(self) -> None:
        observed = {}

        class FakeAsyncClient(_FakeAsyncClient):
            instances = 0
            closed_count = 0

        FakeAsyncClient.observed = observed

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            timeout_seconds=3,
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        async def _run() -> None:
            await client.arecord(LLMUsageEvent(provider="openai", model="gpt-4o-mini", source_event_id="event-1"))
            await client.arecord(LLMUsageEvent(provider="openai", model="gpt-4o-mini", source_event_id="event-2"))
            self.assertTrue(await client.aclose())

        with patch.dict(sys.modules, {"httpx": SimpleNamespace(AsyncClient=FakeAsyncClient)}):
            asyncio.run(_run())

        self.assertEqual(FakeAsyncClient.instances, 1)
        self.assertEqual(FakeAsyncClient.closed_count, 1)
        self.assertEqual(len(observed["contents"]), 2)

    def test_arecord_posts_otlp_json_when_otlp_delivery_mode_is_enabled(self) -> None:
        observed = {}

        class FakeAsyncClient(_FakeAsyncClient):
            instances = 0
            closed_count = 0

        FakeAsyncClient.observed = observed

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            delivery_mode="otlp_http",
            otlp_service_name="checkout-api",
            otlp_service_version="2026.06.1",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with patch.dict(sys.modules, {"httpx": SimpleNamespace(AsyncClient=FakeAsyncClient)}):
            asyncio.run(
                client.arecord(
                    LLMUsageEvent(
                        provider="openai",
                        model="gpt-4o-mini",
                        source_event_id="event-async-otlp",
                        request_id="request-async-otlp",
                        input_tokens=2,
                        output_tokens=4,
                    )
                )
            )

        body = json.loads(observed["content"].decode("utf-8"))
        self.assertEqual(observed["url"], TEST_OTLP_URL)
        self.assertEqual(observed["headers"]["authorization"], "Bearer pat-test")
        span = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attrs = {attribute["key"]: attribute["value"] for attribute in span["attributes"]}
        self.assertEqual(attrs["source_event_id"]["stringValue"], "event-async-otlp")
        self.assertEqual(attrs["gen_ai.request.id"]["stringValue"], "request-async-otlp")
        self.assertEqual(attrs["gen_ai.usage.total_tokens"]["intValue"], 6)

    def test_arecord_does_not_leak_cloptima_authorization_to_non_cloptima_otlp_hosts(self) -> None:
        observed = {}

        class FakeAsyncClient(_FakeAsyncClient):
            instances = 0
            closed_count = 0

        FakeAsyncClient.observed = observed

        client = CloptimaLLMObservability(
            api_base_url="http://127.0.0.1:4318",
            api_key="pat-test",
            delivery_mode="otlp_http",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with patch.dict(sys.modules, {"httpx": SimpleNamespace(AsyncClient=FakeAsyncClient)}):
            asyncio.run(client.arecord(LLMUsageEvent(provider="openai", model="gpt-4o-mini", source_event_id="event-async-otlp-local")))

        self.assertEqual(observed["url"], "http://127.0.0.1:4318/v1/ai/integrations/otlp/traces")
        self.assertNotIn("authorization", observed["headers"])

    def test_record_async_uses_bounded_worker_and_batches_events(self) -> None:
        observed_bodies = []

        def fake_urlopen(request, timeout):
            observed_bodies.append(json.loads(request.data.decode("utf-8")))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_batch_size=10,
            async_flush_interval_seconds=0.05,
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            client.record_async(LLMUsageEvent(provider="openai", model="gpt-4o-mini", source_event_id="event-1"))
            client.record_async(LLMUsageEvent(provider="anthropic", model="claude-3-5-sonnet", source_event_id="event-2"))
            self.assertTrue(client.flush(timeout_seconds=2))
            self.assertTrue(client.close(timeout_seconds=2))

        self.assertEqual(len(observed_bodies), 1)
        self.assertEqual(observed_bodies[0]["schema_version"], "cloptima.llm.batch.v1")
        self.assertEqual(observed_bodies[0]["batch_schema_version"], "cloptima.llm.batch.v1")
        self.assertEqual(observed_bodies[0]["sdk_delivery_stats"]["delivered_events"], 0)
        self.assertEqual(observed_bodies[0]["events"][0]["schema_version"], "cloptima.llm.event.v1")
        self.assertEqual([event["source_event_id"] for event in observed_bodies[0]["events"]], ["event-1", "event-2"])
        self.assertEqual(client.stats().delivered_events, 2)

    def test_record_async_flush_and_close_honor_strict_timeouts(self) -> None:
        def fake_urlopen(_request, timeout=None):
            time.sleep(0.2)
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_flush_interval_seconds=0,
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            client.record_async(LLMUsageEvent(provider="openai", model="gpt-4o-mini", source_event_id="event-timeout"))
            self.assertFalse(client.flush(timeout_seconds=0.01))
            self.assertFalse(client.close(timeout_seconds=0.01))
            time.sleep(0.25)

        self.assertFalse(client._worker.is_alive())
        self.assertEqual(client.stats().delivered_events, 1)

    def test_record_async_reports_queue_overflow_and_closed_client_stats(self) -> None:
        errors = []
        drops = []
        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_queue_max_size=1,
            async_flush_interval_seconds=1,
            on_error=errors.append,
            on_drop=lambda event, reason: drops.append((event.source_event_id, reason)),
        )
        client._ensure_worker = lambda: None

        client.record_async(LLMUsageEvent(provider="openai", model="gpt-4o-mini", source_event_id="event-1"))
        client.record_async(LLMUsageEvent(provider="openai", model="gpt-4o-mini", source_event_id="event-2"))
        self.assertIn("queue is full", str(errors[0]))
        self.assertEqual(drops[0], ("event-2", "queue_full"))
        self.assertEqual(client.stats().dropped_events, 1)

        self.assertTrue(client.close(timeout_seconds=2))
        client.record_async(LLMUsageEvent(provider="openai", model="gpt-4o-mini", source_event_id="event-3"))
        self.assertIn("client is closed", str(errors[1]))
        self.assertEqual(drops[1], ("event-3", "client_closed"))
        self.assertEqual(client.stats().dropped_events, 2)

    def test_record_async_reports_background_delivery_failures(self) -> None:
        errors = []

        def failing_urlopen(_request, timeout=None):
            raise urllib.error.URLError("otlp endpoint unavailable")

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_flush_interval_seconds=0,
            on_error=errors.append,
        )

        with patch.object(urllib.request, "urlopen", failing_urlopen):
            client.record_async(LLMUsageEvent(provider="openai", model="gpt-4o-mini", source_event_id="event-fail-1"))
            self.assertTrue(client.flush(timeout_seconds=2))
            self.assertTrue(client.close(timeout_seconds=2))

        self.assertEqual(client.stats().failed_batches, 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("otlp endpoint unavailable", str(errors[0]))

    def test_observe_async_awaits_call_and_records_usage(self) -> None:
        observed = {}

        class FakeAsyncClient(_FakeAsyncClient):
            pass

        FakeAsyncClient.observed = observed

        async def _call():
            return {
                "id": "chatcmpl-async",
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 13,
                    "total_tokens": 24,
                },
            }

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with patch.dict(sys.modules, {"httpx": SimpleNamespace(AsyncClient=FakeAsyncClient)}):
            response = asyncio.run(
                client.observe_async(
                    provider="openai",
                    model="gpt-4o-mini",
                    fire_and_forget=False,
                    call=_call,
                    extract_usage=extract_openai_usage,
                )
            )

        body = json.loads(observed["content"].decode("utf-8"))
        self.assertEqual(response["id"], "chatcmpl-async")
        self.assertEqual(body["provider_request_id"], "chatcmpl-async")
        self.assertEqual(body["input_tokens"], 11)
        self.assertEqual(body["output_tokens"], 13)

    def test_observe_async_preserves_vendor_reported_cost_from_custom_extractor(self) -> None:
        observed = {}

        class FakeAsyncClient(_FakeAsyncClient):
            pass

        FakeAsyncClient.observed = observed

        async def _call():
            return {
                "response": {"id": "resp-async-cost-1", "model": "gemini-2.5-pro"},
                "billing": {"cost_usd": "0.6543"},
            }

        extractor = create_mapped_usage_extractor(
            defaults={"provider": "vertex_ai"},
            fields={
                "provider_request_id": "response.id",
                "model": "response.model",
                "vendor_reported_cost_usd": "billing.cost_usd",
            },
        )

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with patch.dict(sys.modules, {"httpx": SimpleNamespace(AsyncClient=FakeAsyncClient)}):
            asyncio.run(
                client.observe_async(
                    provider="vertex_ai",
                    model="gemini-2.5-pro",
                    fire_and_forget=False,
                    call=_call,
                    extract_usage=extractor,
                )
            )

        body = json.loads(observed["content"].decode("utf-8"))
        self.assertEqual(body["provider_request_id"], "resp-async-cost-1")
        self.assertEqual(body["vendor_reported_cost_usd"], 0.6543)

    def test_observe_async_rejects_async_generators_and_points_to_stream_api(self) -> None:
        class FakeAsyncClient(_FakeAsyncClient):
            pass

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        async def _stream():
            yield {"delta": "hello"}

        with patch.dict(sys.modules, {"httpx": SimpleNamespace(AsyncClient=FakeAsyncClient)}):
            with self.assertRaisesRegex(TypeError, "observe_async_stream"):
                asyncio.run(
                    client.observe_async(
                        provider="openai",
                        model="gpt-4o-mini",
                        fire_and_forget=False,
                        call=lambda: _stream(),
                    )
                )

    def test_observe_async_fire_and_forget_uses_bounded_worker(self) -> None:
        observed_bodies = []

        def fake_urlopen(request, timeout):
            observed_bodies.append(json.loads(request.data.decode("utf-8")))
            return _FakeResponse()

        async def _call():
            return {
                "id": "chatcmpl-async-queued",
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                },
            }

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_batch_size=10,
            async_flush_interval_seconds=0.05,
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            response = asyncio.run(
                client.observe_async(
                    provider="openai",
                    model="gpt-4o-mini",
                    call=_call,
                    extract_usage=extract_openai_usage,
                )
            )
            self.assertTrue(client.flush(timeout_seconds=2))
            self.assertTrue(client.close(timeout_seconds=2))

        self.assertEqual(response["id"], "chatcmpl-async-queued")
        self.assertEqual(len(observed_bodies), 1)
        self.assertEqual(observed_bodies[0]["provider_request_id"], "chatcmpl-async-queued")
        self.assertEqual(observed_bodies[0]["input_tokens"], 3)
        self.assertEqual(observed_bodies[0]["output_tokens"], 5)

    def test_ainstrument_openai_compatible_response_records_existing_async_response(self) -> None:
        observed = {}

        class FakeAsyncClient(_FakeAsyncClient):
            pass

        FakeAsyncClient.observed = observed

        async def _response():
            return {
                "id": "chatcmpl-helper-async",
                "model": "gpt-4o-mini",
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 4,
                    "total_tokens": 13,
                },
            }

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with patch.dict(sys.modules, {"httpx": SimpleNamespace(AsyncClient=FakeAsyncClient)}):
            response = asyncio.run(
                ainstrument_openai_compatible_response(
                    client,
                    _response(),
                    fire_and_forget=False,
                    metadata={"integration_mode": "passive_helper_async"},
                )
            )

        body = json.loads(observed["content"].decode("utf-8"))
        self.assertEqual(response["id"], "chatcmpl-helper-async")
        self.assertEqual(body["provider_request_id"], "chatcmpl-helper-async")
        self.assertEqual(body["input_tokens"], 9)
        self.assertEqual(body["output_tokens"], 4)
        self.assertEqual(body["metadata"]["integration_mode"], "passive_helper_async")
        self.assertGreaterEqual(body["latency_ms"], 0)

    def test_ainstrument_openai_compatible_response_does_not_fabricate_latency_for_existing_responses(self) -> None:
        observed = {}

        class FakeAsyncClient(_FakeAsyncClient):
            pass

        FakeAsyncClient.observed = observed

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with patch.dict(sys.modules, {"httpx": SimpleNamespace(AsyncClient=FakeAsyncClient)}):
            response = asyncio.run(
                ainstrument_openai_compatible_response(
                    client,
                    {
                        "id": "chatcmpl-helper-async-existing",
                        "model": "gpt-4o-mini",
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 2,
                            "total_tokens": 3,
                        },
                    },
                    fire_and_forget=False,
                )
            )

        body = json.loads(observed["content"].decode("utf-8"))
        self.assertEqual(response["id"], "chatcmpl-helper-async-existing")
        self.assertIsNone(body.get("latency_ms"))

    def test_observe_stream_yields_chunks_and_records_final_usage(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        chunks = [
            {"id": "chatcmpl-stream", "model": "gpt-4o-mini", "choices": [{"delta": {"content": "hi"}}]},
            {
                "id": "chatcmpl-stream",
                "model": "gpt-4o-mini",
                "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            },
        ]

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            emitted = list(
                client.observe_stream(
                    provider="openai",
                    model="gpt-4o-mini",
                    fire_and_forget=False,
                    call=lambda: iter(chunks),
                    extract_usage=extract_openai_stream_usage,
                )
            )

        self.assertEqual(emitted, chunks)
        self.assertEqual(observed["body"]["status"], "succeeded")
        self.assertEqual(observed["body"]["provider_request_id"], "chatcmpl-stream")
        self.assertEqual(observed["body"]["input_tokens"], 5)
        self.assertEqual(observed["body"]["output_tokens"], 7)
        self.assertEqual(observed["body"]["metadata"]["streamed"], True)

    def test_observe_stream_preserves_vendor_reported_cost_from_custom_extractor(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        chunks = [
            {"delta": "frame-1"},
            {"response": {"id": "resp-stream-cost-1", "model": "veo-3"}},
            {"billing": {"cost_usd": "1.2345"}},
        ]

        extractor = create_mapped_usage_extractor(
            defaults={"provider": "vertex_ai"},
            fields={
                "provider_request_id": ["response.id"],
                "model": ["response.model"],
                "vendor_reported_cost_usd": ["billing.cost_usd"],
            },
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            emitted = list(
                client.observe_stream(
                    provider="vertex_ai",
                    model="veo-3",
                    fire_and_forget=False,
                    call=lambda: iter(chunks),
                    extract_usage=lambda items: extractor(items[-1] if items else {}),
                )
            )

        self.assertEqual(emitted, chunks)
        self.assertEqual(observed["body"]["vendor_reported_cost_usd"], 1.2345)
        self.assertEqual(observed["body"]["metadata"]["streamed"], True)

    def test_instrument_openai_compatible_stream_observes_existing_stream(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        chunks = [
            {"id": "chatcmpl-helper-stream", "model": "gpt-4o-mini", "choices": [{"delta": {"content": "hi"}}]},
            {
                "id": "chatcmpl-helper-stream",
                "model": "gpt-4o-mini",
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
        ]

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            emitted = list(
                instrument_openai_compatible_stream(
                    client,
                    iter(chunks),
                    fire_and_forget=False,
                    metadata={"integration_mode": "passive_helper_stream"},
                )
            )

        self.assertEqual(emitted, chunks)
        self.assertEqual(observed["body"]["provider_request_id"], "chatcmpl-helper-stream")
        self.assertEqual(observed["body"]["input_tokens"], 2)
        self.assertEqual(observed["body"]["output_tokens"], 3)
        self.assertEqual(observed["body"]["metadata"]["integration_mode"], "passive_helper_stream")

    def test_instrument_openai_compatible_stream_preserves_azure_openai_provider_attribution(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse()

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        chunks = [
            {"id": "azure-stream-1", "model": "gpt-4o-mini"},
            {
                "id": "azure-stream-1",
                "model": "gpt-4o-mini",
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
        ]

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            emitted = list(
                instrument_openai_compatible_stream(
                    client,
                    iter(chunks),
                    provider="azure_openai",
                    fire_and_forget=False,
                )
            )

        self.assertEqual(emitted, chunks)
        self.assertEqual(observed["body"]["provider"], "azure_openai")

    def test_observe_stream_records_partial_on_interrupted_stream(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse()

        def interrupted_stream():
            yield {"delta": "first"}
            raise RuntimeError("stream interrupted")

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            with self.assertRaisesRegex(RuntimeError, "stream interrupted"):
                list(
                    client.observe_stream(
                        provider="anthropic",
                        model="claude-3-5-sonnet",
                        fire_and_forget=False,
                        call=interrupted_stream,
                    )
                )

        self.assertEqual(observed["body"]["status"], "partial")
        self.assertEqual(observed["body"]["metadata"]["streamed"], True)
        self.assertEqual(observed["body"]["metadata"]["stream_chunks"], 1)

    def test_observe_async_stream_fire_and_forget_uses_bounded_worker(self) -> None:
        observed_bodies = []

        def fake_urlopen(request, timeout):
            observed_bodies.append(json.loads(request.data.decode("utf-8")))
            return _FakeResponse()

        async def async_chunks():
            yield {"id": "chatcmpl-async-stream", "model": "gpt-4o-mini"}
            yield {
                "id": "chatcmpl-async-stream",
                "model": "gpt-4o-mini",
                "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
            }

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_batch_size=10,
            async_flush_interval_seconds=0.05,
        )

        async def _run():
            return [
                chunk
                async for chunk in client.observe_async_stream(
                    provider="openai",
                    model="gpt-4o-mini",
                    call=async_chunks,
                    extract_usage=extract_openai_stream_usage,
                    fire_and_forget=True,
                )
            ]

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            emitted = asyncio.run(_run())
            self.assertTrue(client.flush(timeout_seconds=2))
            self.assertTrue(client.close(timeout_seconds=2))

        self.assertEqual(len(emitted), 2)
        self.assertEqual(len(observed_bodies), 1)
        self.assertEqual(observed_bodies[0]["provider_request_id"], "chatcmpl-async-stream")
        self.assertEqual(observed_bodies[0]["metadata"]["streamed"], True)
        self.assertEqual(observed_bodies[0]["input_tokens"], 2)
        self.assertEqual(observed_bodies[0]["output_tokens"], 4)

    def test_ainstrument_openai_compatible_stream_observes_existing_async_stream(self) -> None:
        observed = {}

        class FakeAsyncClient(_FakeAsyncClient):
            pass

        FakeAsyncClient.observed = observed

        async def async_chunks():
            yield {"id": "chatcmpl-helper-async-stream", "model": "gpt-4o-mini"}
            yield {
                "id": "chatcmpl-helper-async-stream",
                "model": "gpt-4o-mini",
                "usage": {"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 7},
            }

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        async def _run():
            return [
                chunk
                async for chunk in ainstrument_openai_compatible_stream(
                    client,
                    async_chunks(),
                    metadata={"integration_mode": "passive_helper_async_stream"},
                )
            ]

        with patch.dict(sys.modules, {"httpx": SimpleNamespace(AsyncClient=FakeAsyncClient)}):
            emitted = asyncio.run(_run())

        self.assertEqual(len(emitted), 2)
        body = json.loads(observed["content"].decode("utf-8"))
        self.assertEqual(body["provider_request_id"], "chatcmpl-helper-async-stream")
        self.assertEqual(body["input_tokens"], 6)
        self.assertEqual(body["output_tokens"], 1)
        self.assertEqual(body["metadata"]["integration_mode"], "passive_helper_async_stream")

    def test_observe_async_stream_defaults_to_synchronous_recording(self) -> None:
        observed = {}

        class FakeAsyncClient(_FakeAsyncClient):
            pass

        FakeAsyncClient.observed = observed

        async def async_chunks():
            yield {"id": "chatcmpl-sync-stream", "model": "gpt-4o-mini"}
            yield {
                "id": "chatcmpl-sync-stream",
                "model": "gpt-4o-mini",
                "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            }

        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        async def _run():
            return [
                chunk
                async for chunk in client.observe_async_stream(
                    provider="openai",
                    model="gpt-4o-mini",
                    call=async_chunks,
                    extract_usage=extract_openai_stream_usage,
                )
            ]

        with patch.dict(sys.modules, {"httpx": SimpleNamespace(AsyncClient=FakeAsyncClient)}):
            emitted = asyncio.run(_run())

        self.assertEqual(len(emitted), 2)
        body = json.loads(observed["content"].decode("utf-8"))
        self.assertEqual(body["provider_request_id"], "chatcmpl-sync-stream")
        self.assertEqual(body["input_tokens"], 3)
        self.assertIsNone(client._worker)

    def test_request_context_helpers_extract_passive_http_metadata(self) -> None:
        fastapi_request = SimpleNamespace(
            headers={
                "x-request-id": "req-fastapi-1",
                "x-trace-id": "trace-fastapi-1",
                "host": "api.example.com",
                "user-agent": "pytest",
                "x-org": "growth",
            },
            method="POST",
            url=SimpleNamespace(path="/v1/chat", netloc="api.example.com"),
            client=SimpleNamespace(host="10.0.0.1"),
        )
        flask_request = SimpleNamespace(
            headers={
                "x-request-id": "req-flask-1",
                "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00",
                "host": "app.example.com",
                "user-agent": "pytest",
            },
            method="PATCH",
            path="/api/ai",
            host="app.example.com",
            remote_addr="10.0.0.2",
            url_rule=SimpleNamespace(rule="/api/ai"),
        )
        httpx_request = SimpleNamespace(
            headers={
                "x-request-id": "req-httpx-1",
                "host": "provider.example.com",
                "user-agent": "httpx-test",
            },
            method="GET",
            url=SimpleNamespace(host="provider.example.com", path="/v1/chat", __str__=lambda self: "https://provider.example.com/v1/chat"),
        )

        fastapi_context = instrument_fastapi_request_context(
            fastapi_request,
            attribution={"team_id": "platform"},
            include_headers=["x-org"],
        )
        flask_context = instrument_flask_request_context(flask_request)
        httpx_context = instrument_httpx_transport_metadata(httpx_request)

        self.assertEqual(fastapi_context["request_id"], "req-fastapi-1")
        self.assertEqual(fastapi_context["trace_id"], "trace-fastapi-1")
        self.assertEqual(fastapi_context["metadata"]["http_method"], "POST")
        self.assertEqual(fastapi_context["metadata"]["http_path"], "/v1/chat")
        self.assertEqual(fastapi_context["metadata"]["http_header_x_org"], "growth")
        self.assertEqual(fastapi_context["attribution"]["team_id"], "platform")

        self.assertEqual(flask_context["request_id"], "req-flask-1")
        self.assertEqual(flask_context["trace_id"], "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00")
        self.assertEqual(flask_context["metadata"]["http_method"], "PATCH")
        self.assertEqual(flask_context["metadata"]["http_route"], "/api/ai")

        self.assertEqual(httpx_context["request_id"], "req-httpx-1")
        self.assertEqual(httpx_context["metadata"]["http_host"], "provider.example.com")
        self.assertEqual(httpx_context["metadata"]["http_path"], "/v1/chat")
        self.assertIn("provider.example.com", httpx_context["metadata"]["provider_endpoint"])

    def test_instrument_httpx_client_records_usage_for_openai_compatible_responses(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["request"] = request
            return _FakeResponse()

        provider_request = SimpleNamespace(
            headers={"x-request-id": "req-httpx-2", "user-agent": "httpx-test"},
            method="POST",
            url=SimpleNamespace(
                host="api.openai.com",
                path="/v1/chat/completions",
                __str__=lambda self: "https://api.openai.com/v1/chat/completions",
            ),
        )
        response = _FakeHttpxResponse(
            {
                "id": "chatcmpl-httpx-1",
                "model": "gpt-4o-mini",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 7,
                    "total_tokens": 19,
                },
            },
            headers={"openai-request-id": "chatcmpl-httpx-1"},
            request=provider_request,
        )
        httpx_client = _FakeHttpxClient(response=response)
        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            wrapped = instrument_httpx_client(
                httpx_client,
                cloptima=client,
                provider="openai",
                model="gpt-4o-mini",
                metadata={"integration_mode": "httpx_client"},
                fire_and_forget=False,
            )
            result = wrapped.request("POST", "https://api.openai.com/v1/chat/completions")

        self.assertIs(result, response)
        body = json.loads(observed["request"].data.decode("utf-8"))
        self.assertEqual(body["provider"], "openai")
        self.assertEqual(body["model"], "gpt-4o-mini")
        self.assertEqual(body["provider_request_id"], "chatcmpl-httpx-1")
        self.assertEqual(body["input_tokens"], 12)
        self.assertEqual(body["output_tokens"], 7)
        self.assertEqual(body["metadata"]["integration_mode"], "httpx_client")
        self.assertEqual(body["metadata"]["http_method"], "POST")
        self.assertEqual(body["metadata"]["http_path"], "/v1/chat/completions")
        self.assertEqual(body["metadata"]["http_status_code"], 200)
        self.assertTrue(body["metadata"]["response_json_parsed"])

    def test_instrument_httpx_client_fails_open_when_provider_is_missing(self) -> None:
        errors = []
        observed = {"posts": 0}
        httpx_client = _FakeHttpxClient(response=_FakeHttpxResponse({}, status_code=204))
        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        def fake_urlopen(request, timeout):
            observed["posts"] += 1
            return _FakeResponse()

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            wrapped = instrument_httpx_client(
                httpx_client,
                cloptima=client,
                on_instrumentation_error=errors.append,
                fire_and_forget=False,
            )
            result = wrapped.request("GET", "https://provider.example.com/v1/chat")

        self.assertEqual(result.status_code, 204)
        self.assertEqual(len(httpx_client.calls), 1)
        self.assertEqual(observed["posts"], 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("requires a provider", str(errors[0]))

    def test_instrument_httpx_transport_records_network_failures(self) -> None:
        observed = {}

        def fake_urlopen(request, timeout):
            observed["request"] = request
            return _FakeResponse()

        request = SimpleNamespace(
            headers={"x-request-id": "req-httpx-transport-1"},
            method="POST",
            url=SimpleNamespace(
                host="api.openai.com",
                path="/v1/responses",
                __str__=lambda self: "https://api.openai.com/v1/responses",
            ),
        )
        transport = _FakeHttpxTransport(error=RuntimeError("connection reset"))
        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            wrapped = instrument_httpx_transport(
                transport,
                cloptima=client,
                provider="openai",
                model="gpt-4o-mini",
                metadata={"integration_mode": "httpx_transport"},
                fire_and_forget=False,
            )
            with self.assertRaisesRegex(RuntimeError, "connection reset"):
                wrapped.handle_request(request)

        body = json.loads(observed["request"].data.decode("utf-8"))
        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["request_id"], "req-httpx-transport-1")
        self.assertEqual(body["metadata"]["integration_mode"], "httpx_transport")
        self.assertEqual(body["metadata"]["http_path"], "/v1/responses")
        self.assertEqual(body["error_message"], "connection reset")

    def test_instrument_httpx_client_preserves_sync_context_manager_behavior(self) -> None:
        httpx_client = _FakeHttpxClient(response=_FakeHttpxResponse({"ok": True}))
        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
        )

        wrapped = instrument_httpx_client(
            httpx_client,
            cloptima=client,
            provider="openai",
            model="gpt-4o-mini",
        )

        with wrapped as entered:
            self.assertIs(entered, wrapped)

        self.assertEqual(httpx_client.entered, 1)
        self.assertEqual(httpx_client.exited, 1)

    def test_instrument_httpx_client_supports_async_clients(self) -> None:
        provider_request = SimpleNamespace(
            headers={"x-request-id": "req-httpx-async-1"},
            method="POST",
            url=SimpleNamespace(
                host="api.anthropic.com",
                path="/v1/messages",
                __str__=lambda self: "https://api.anthropic.com/v1/messages",
            ),
        )
        response = _FakeHttpxResponse(
            {
                "id": "msg_123",
                "model": "claude-3-5-sonnet",
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
            headers={"anthropic-request-id": "msg_123"},
            request=provider_request,
        )
        httpx_client = _FakeAsyncHttpxClient(response=response)
        _FakeAsyncClient.observed = {}
        ingest_client = _FakeAsyncClient(timeout=3.0)
        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_http_client=ingest_client,
        )

        async def run_test():
            wrapped = instrument_httpx_client(
                httpx_client,
                cloptima=client,
                provider="anthropic",
                model="claude-3-5-sonnet",
                metadata={"integration_mode": "httpx_async_client"},
                fire_and_forget=False,
            )
            return await wrapped.request("POST", "https://api.anthropic.com/v1/messages")

        result = asyncio.run(run_test())

        self.assertIs(result, response)
        body = json.loads(_FakeAsyncClient.observed["content"].decode("utf-8"))
        self.assertEqual(body["provider"], "anthropic")
        self.assertEqual(body["provider_request_id"], "msg_123")
        self.assertEqual(body["input_tokens"], 5)
        self.assertEqual(body["output_tokens"], 3)
        self.assertEqual(body["metadata"]["integration_mode"], "httpx_async_client")

    def test_instrument_httpx_client_preserves_async_context_manager_behavior(self) -> None:
        httpx_client = _FakeAsyncHttpxClient(response=_FakeHttpxResponse({"ok": True}))
        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_http_client=_FakeAsyncClient(timeout=3.0),
        )

        async def run_test():
            wrapped = instrument_httpx_client(
                httpx_client,
                cloptima=client,
                provider="anthropic",
                model="claude-3-5-sonnet",
            )
            async with wrapped as entered:
                self.assertIs(entered, wrapped)

        asyncio.run(run_test())

        self.assertEqual(httpx_client.entered, 1)
        self.assertEqual(httpx_client.exited, 1)

    def test_arecord_uses_custom_async_client_without_importing_httpx(self) -> None:
        observed = {}
        custom_async_client = _FakeAsyncClient(timeout=3.0)
        custom_async_client.observed = observed
        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_http_client=custom_async_client,
        )

        original_import = builtins.__import__

        def blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "httpx":
                raise AssertionError("httpx should not be imported when a reusable async client is provided")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=blocking_import):
            asyncio.run(
                client.arecord(
                    LLMUsageEvent(
                        provider="openai",
                        model="gpt-4o-mini",
                        request_id="req-custom-async-client-1",
                        input_tokens=2,
                        output_tokens=1,
                    )
                )
            )

        body = json.loads(observed["content"].decode("utf-8"))
        self.assertEqual(body["request_id"], "req-custom-async-client-1")
        self.assertEqual(body["input_tokens"], 2)
        self.assertEqual(body["output_tokens"], 1)

    def test_arecord_rejects_closed_external_async_client_without_importing_httpx(self) -> None:
        custom_async_client = _FakeAsyncClient(timeout=3.0)
        custom_async_client.is_closed = True
        client = CloptimaLLMObservability(
            api_base_url=TEST_API_BASE_URL,
            api_key="pat-test",
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev"),
            async_http_client=custom_async_client,
        )

        original_import = builtins.__import__

        def blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "httpx":
                raise AssertionError("httpx should not be imported for a closed external async client")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=blocking_import):
            with self.assertRaisesRegex(RuntimeError, "provided async_http_client is closed"):
                asyncio.run(
                    client.arecord(
                        LLMUsageEvent(
                            provider="openai",
                            model="gpt-4o-mini",
                            request_id="req-closed-custom-async-client-1",
                            input_tokens=2,
                            output_tokens=1,
                        )
                    )
                )

    def test_preview_helpers_build_sanitized_payloads_and_otlp_dry_runs(self) -> None:
        payload = preview_event_payload(
            LLMUsageEvent(
                provider="openai",
                model="gpt-4o-mini",
                request_id="req-preview-1",
                metadata={"route": "/chat", "prompt": "secret prompt"},
            ),
            default_attribution=LLMAttribution(app_id="agent-api", environment="dev", team_id="platform"),
            metadata_policy={"mode": "allowlisted_metadata", "allowlist_keys": ["route"]},
            sdk_version="0.1.0",
        )
        validation = validate_payload(payload)
        otlp = preview_otlp_request(payload, sdk_version="0.1.0", service_name="agent-api")

        self.assertTrue(validation["valid"])
        self.assertEqual(validation["errors"], [])
        self.assertEqual(payload["schema_version"], "cloptima.llm.event.v1")
        self.assertEqual(payload["metadata"]["app_id"], "agent-api")
        self.assertEqual(payload["metadata"]["route"], "/chat")
        self.assertNotIn("prompt", payload["metadata"])
        self.assertEqual(
            otlp["resourceSpans"][0]["resource"]["attributes"][0]["value"]["stringValue"],
            "agent-api",
        )
        self.assertEqual(
            otlp["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["name"],
            "llm.openai.gpt-4o-mini",
        )

        batch_payload = preview_batch_payload(
            [
                LLMUsageEvent(provider="openai", model="gpt-4o-mini", source_event_id="evt-1"),
                LLMUsageEvent(provider="anthropic", model="claude-3-5-sonnet", source_event_id="evt-2"),
            ],
            default_attribution={"app_id": "agent-api", "environment": "dev"},
        )
        self.assertEqual(batch_payload["schema_version"], "cloptima.llm.batch.v1")
        self.assertEqual([event["source_event_id"] for event in batch_payload["events"]], ["evt-1", "evt-2"])

    def test_preview_helpers_accept_plain_event_mappings_with_flat_attribution_fields(self) -> None:
        payload = preview_event_payload(
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "request_id": "req-preview-mapping-1",
                "input_tokens": 10,
                "output_tokens": 5,
                "app_id": "agent-api",
                "environment": "dev",
                "feature_id": "customer_summary",
                "metadata": {"route": "/chat"},
            }
        )
        batch_payload = preview_batch_payload(
            [
                {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "source_event_id": "evt-map-1",
                    "app_id": "agent-api",
                    "environment": "dev",
                },
                {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet",
                    "source_event_id": "evt-map-2",
                    "app_id": "agent-api",
                    "environment": "dev",
                },
            ]
        )

        self.assertEqual(payload["metadata"]["app_id"], "agent-api")
        self.assertEqual(payload["metadata"]["environment"], "dev")
        self.assertEqual(payload["metadata"]["feature_id"], "customer_summary")
        self.assertEqual(payload["total_tokens"], 15)
        self.assertEqual([event["source_event_id"] for event in batch_payload["events"]], ["evt-map-1", "evt-map-2"])

    def test_validate_payload_reports_malformed_event_and_batch_payloads(self) -> None:
        invalid = validate_payload(
            {
                "schema_version": "wrong",
                "sdk_name": "",
                "provider": "",
                "model": "",
                "metadata": "bad",
                "status": "maybe",
                "input_tokens": -1,
            }
        )
        self.assertFalse(invalid["valid"])
        self.assertTrue(any("schema_version" in error for error in invalid["errors"]))
        self.assertTrue(any("sdk_name" in error for error in invalid["errors"]))
        self.assertTrue(any("metadata" in error for error in invalid["errors"]))

        invalid_batch = validate_payload({"schema_version": "wrong-batch", "events": ["bad-event"]})
        self.assertFalse(invalid_batch["valid"])
        self.assertTrue(any("batch.schema_version" in error for error in invalid_batch["errors"]))
        self.assertTrue(any("events[0] must be an object" in error for error in invalid_batch["errors"]))

    def test_anthropic_stream_extractor_aggregates_message_events(self) -> None:
        self.assertEqual(
            extract_anthropic_stream_usage(
                [
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg-stream",
                            "model": "claude-3-5-sonnet",
                            "usage": {"input_tokens": 8, "cache_read_input_tokens": 2},
                        },
                    },
                    {"type": "message_delta", "usage": {"output_tokens": 4, "cache_creation_input_tokens": 3}},
                ]
            ),
            {
                "provider": "anthropic",
                "provider_request_id": "msg-stream",
                "model": "claude-3-5-sonnet",
                "input_tokens": 8,
                "output_tokens": 4,
                "total_tokens": 12,
                "cached_input_tokens": 2,
                "extra_usage_units": {"cache_write": 3},
                "cache_hit": True,
            },
        )

    def test_cloud_provider_stream_extractors_normalize_usage(self) -> None:
        gemini = extract_gemini_stream_usage(
            [
                {"responseId": "gemini-stream-1", "modelVersion": "gemini-2.5-pro"},
                {"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 11, "totalTokenCount": 18}},
            ]
        )
        vertex = extract_vertex_stream_usage(
            [
                {"response_id": "vertex-stream-1", "model_version": "gemini-2.5-pro"},
                {"usage_metadata": {"prompt_token_count": 2, "candidates_token_count": 4, "total_token_count": 6}},
            ]
        )
        bedrock = extract_bedrock_stream_usage(
            [
                {"requestId": "bedrock-stream-1", "modelId": "anthropic.claude-3-5-sonnet", "usage": {"inputTokens": 5}},
                {"usage": {"outputTokens": 9}},
            ]
        )

        self.assertEqual(gemini["provider_request_id"], "gemini-stream-1")
        self.assertEqual(gemini["total_tokens"], 18)
        self.assertEqual(vertex["provider"], "vertex_ai")
        self.assertEqual(vertex["total_tokens"], 6)
        self.assertEqual(bedrock["provider_request_id"], "bedrock-stream-1")
        self.assertEqual(bedrock["total_tokens"], 14)

    def test_provider_usage_fixture_replay_covers_supported_sdk_extractors(self) -> None:
        fixture_candidates = [
            Path(__file__).resolve().parents[2] / "llm-observability-fixtures" / "provider_usage_replay.json",
            Path(__file__).resolve().parents[1] / "llm-observability-fixtures" / "provider_usage_replay.json",
        ]
        fixture_path = next((path for path in fixture_candidates if path.exists()), None)
        if fixture_path is None:
            raise FileNotFoundError("provider_usage_replay.json fixture not found")
        fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))

        def extract(fixture):
            provider = fixture["provider"]
            kind = fixture["kind"]
            payload = fixture["payload"]
            if provider == "openai" and kind == "stream":
                return extract_openai_stream_usage(payload)
            if provider == "openai":
                return extract_openai_usage(payload)
            if provider == "azure_openai":
                return extract_azure_openai_usage(payload)
            if provider == "anthropic" and kind == "stream":
                return extract_anthropic_stream_usage(payload)
            if provider == "anthropic":
                return extract_anthropic_usage(payload)
            if provider == "gemini" and kind == "stream":
                return extract_gemini_stream_usage(payload)
            if provider == "gemini":
                return extract_gemini_usage(payload)
            if provider == "vertex_ai" and kind == "stream":
                return extract_vertex_stream_usage(payload)
            if provider == "vertex_ai":
                return extract_vertex_usage(payload)
            if provider == "bedrock" and kind == "stream":
                return extract_bedrock_stream_usage(payload)
            if provider == "bedrock":
                return extract_bedrock_usage(payload)
            raise AssertionError(f"unsupported fixture {fixture['name']}")

        for fixture in fixtures:
            with self.subTest(fixture=fixture["name"]):
                self.assertEqual(extract(fixture), fixture["expected"])


if __name__ == "__main__":
    unittest.main()
