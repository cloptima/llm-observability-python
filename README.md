# Cloptima LLM Observability Python SDK

Capture LLM usage telemetry from your application and send it to Cloptima for cost reporting, attribution, and usage analytics.

This SDK is designed for teams that want observability without replacing their existing provider clients, wrappers, retries, auth, or application security controls.

## Install

```bash
pip install cloptima-llm-observability
```

If you want the `httpx` transport helpers:

```bash
pip install "cloptima-llm-observability[httpx]"
```

## Quick start

Required configuration:

- `CLOPTIMA_LLM_OBSERVABILITY_API_KEY`
- `CLOPTIMA_LLM_OBSERVABILITY_APP_ID`

Recommended while testing:

- `CLOPTIMA_LLM_OBSERVABILITY_ENVIRONMENT=dev`

```python
from cloptima_llm_observability import extract_openai_usage, init_from_env

cloptima = init_from_env()

result = cloptima.observe_call(
    provider="openai",
    model="gpt-4.1-mini",
    call=lambda: summary_service.generate(prompt),
    extract_usage=extract_openai_usage,
    feature_id="summary_generation",
    workflow_id="support_agent",
    fire_and_forget=False,
)
```

By default, the SDK sends bearer-authenticated HTTPS requests to Cloptima at `https://api.cloptima.ai/v1/ai/integrations/sdk/events`.

If the required configuration is missing, `init_from_env()` returns a disabled pass-through client so local development and tests do not break.

## Choose your integration path

### Call-site or wrapper boundary

This is the default path for most teams.

Use it when you already know the provider, model, and business context at the point where your code calls an LLM or an existing AI wrapper.

- `observe_call(...)` for direct integration
- `create_observed_call(...)` for reusable wrappers
- `wrap_observed_service(...)` to instrument customer-owned service classes

```python
from cloptima_llm_observability import (
    extract_openai_usage,
    init_from_env,
    wrap_observed_service,
)


class SummaryService:
    def generate_summary(self, prompt: str):
        return openai.responses.create(model="gpt-4.1-mini", input=prompt)


cloptima = init_from_env()
summary_service = wrap_observed_service(
    cloptima,
    SummaryService(),
    {
        "generate_summary": {
            "kind": "call",
            "options": {
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "extract_usage": extract_openai_usage,
                "fire_and_forget": False,
            },
            "resolve_overrides": lambda prompt: {
                "attribution": {
                    "feature_id": "summary_generation",
                },
            },
        }
    },
)
```

### Context-first attribution

Use context helpers when you want workflow or feature attribution to apply across nested calls without threading more parameters through your own service signatures.

- `with_attribution(...)`
- `run_with_attribution(...)`
- `with_workflow(...)`
- `with_task(...)`
- `@workflow(...)`
- `@task(...)`

```python
from cloptima_llm_observability import with_task, with_workflow

with with_workflow("support_agent", tenant_id="acme-prod"):
    with with_task("draft_reply", team_id="customer-support"):
        summary_service.generate_summary(prompt)
```

Per-call attribution still works and overrides context when needed.

### Shared transport integration

If your application centralizes outbound LLM calls behind `httpx`, instrument that shared boundary:

```python
import httpx

from cloptima_llm_observability import init_from_env, instrument_httpx_transport

cloptima = init_from_env()
transport = instrument_httpx_transport(
    httpx.HTTPTransport(),
    cloptima=cloptima,
    provider="openai",
    model="gpt-4o-mini",
    fire_and_forget=False,
)
client = httpx.Client(transport=transport)
```

This gives broad coverage, but it has less business context than call-site or wrapper-boundary integration.

### OTLP delivery to Cloptima

Use `otlp_http` when your enterprise prefers OpenTelemetry-compatible payloads but still wants to send that telemetry to Cloptima.

- `cloptima_http` is the default delivery mode
- `otlp_http` sends OpenTelemetry-compatible payloads to Cloptima's OTLP receiver

```bash
CLOPTIMA_LLM_OBSERVABILITY_DELIVERY_MODE=otlp_http
CLOPTIMA_LLM_OBSERVABILITY_OTLP_SERVICE_NAME=agent-api
CLOPTIMA_LLM_OBSERVABILITY_OTLP_SERVICE_VERSION=2026.06.14
```

If you already operate an OTEL collector and emit GenAI spans, you can also send OTLP data to Cloptima without using this SDK. Use the SDK OTLP mode when you want application-managed instrumentation that still fits an OTLP-shaped delivery contract.

## Built-in extractors and compatibility

Built-in usage extractors cover:

- OpenAI
- Azure OpenAI
- Anthropic
- Gemini
- Vertex AI
- Bedrock

If a provider reports image, audio, or video token usage, the built-in extractors capture those units in fields such as `input_image`, `output_image`, `input_audio`, and `output_video`. When Cloptima has pricing for that model, those units can be included in cost reporting.

If a provider returns a direct charge, pass or preserve it as `vendor_reported_cost_usd`.

The SDK does not invent media charges for providers that bill by image count, video duration, resolution, or other non-token measures when the provider response does not expose enough pricing data. In those cases, either:

- preserve the provider-reported cost when available
- map the provider's usage fields into `extra_usage_units`
- or add your own custom extractor until the provider exposes a stable shape

If a provider response shape drifts, you do not need to replace the whole extractor path. Compose or patch it instead:

- `try_extract_usage(...)`
- `compose_usage_extractors(...)`
- `with_usage_overrides(...)`
- `create_mapped_usage_extractor(...)`
- `list_supported_providers()`

Example:

```python
from cloptima_llm_observability import create_mapped_usage_extractor, init_from_env

cloptima = init_from_env()

extract_usage = create_mapped_usage_extractor(
    defaults={
        "provider": "gemini",
    },
    fields={
        "model": "modelVersion",
        "provider_request_id": "responseId",
        "vendor_reported_cost_usd": "billing.costUsd",
    },
    number_fields={
        "input_tokens": "usage.promptTokenCount",
        "output_tokens": "usage.responseTokenCount",
        "total_tokens": "usage.totalTokenCount",
    },
    extra_usage_units={
        "output_image": "usage.outputImageTokenCount",
    },
)
```

## Attribution fields

Common ownership and reporting fields:

- `app_id`
- `environment`
- `team_id`
- `feature_id`
- `workflow_id`
- `cost_center`
- `business_unit`
- `product`
- `tenant_id`
- `end_customer_id`
- `customer_segment`
- `release`

Set defaults once in `default_attribution`, set them in context, or override them per call.

## Metadata and privacy

Use `metadata_policy` to control how custom metadata is retained:

- `metadata_only`
- `allowlisted_metadata`
- `strict_finops`
- `debug_observability`

Sensitive-looking keys such as prompts, messages, credentials, and secrets are treated conservatively by default.

## Validation and local previews

Use these helpers in local tests, CI, or rollout checks:

- `preview_event_payload(...)`
- `preview_batch_payload(...)`
- `preview_otlp_request(...)`
- `validate_payload(...)`

They build or validate payloads in memory and do not send network traffic.

## Examples

Public examples live in `examples/`:

- `basic.py`: direct call-site integration
- `custom_wrapper.py`: existing service wrapper integration
- `workflow_context.py`: context-first attribution without signature bloat
- `httpx_transport.py`: shared `httpx` integration
- `multimodal_tokens.py`: token-based multimodal usage extraction for image, audio, and video inputs and outputs
- `mapped_extractor.py`: adapt a provider or internal wrapper response without rewriting your integration
- `otlp_basic.py`: OTLP-compatible delivery to Cloptima
- `openai_basic.py`, `anthropic_basic.py`, `gemini_basic.py`: provider-specific extractor examples

## Troubleshooting

No telemetry arrives:

- verify the API key is valid for Cloptima telemetry ingestion
- check `client.is_enabled()`
- inspect a sample event with `validate_payload(preview_event_payload(...))`

Unexpected provider response shape:

- start with the closest built-in extractor
- patch field differences with `with_usage_overrides(...)` or `create_mapped_usage_extractor(...)`
- compare against `list_supported_providers()` if you need a supported-provider snapshot

## Support

- Issues: `https://github.com/cloptima/cloptima-llm-observability-python/issues`
- Security: see `SECURITY.md`
- Product support: `hello@cloptima.ai`
