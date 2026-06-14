# Changelog

## 0.2.0

- Added context-first attribution helpers so workflows and features can be set once and inherited across nested LLM calls.
- Added wrapper-boundary helpers for teams that already route LLM access through internal service classes.
- Added more resilient extractor composition and field-mapping helpers for provider SDK drift and custom wrapper payloads.
- Expanded multimodal usage support for token-based image, audio, and video inputs and outputs across supported providers.
- Preserved provider-reported cost values when extractors return them, so direct provider billing data can flow through to Cloptima.
- Added stronger public examples for multimodal usage, mapped extractors, OTLP delivery, and context-based attribution.

## 0.1.3

- Improved telemetry ingestion request headers to resolve potential network connectivity and compatibility issues in some network environments.

## 0.1.2

- Added stronger public examples for existing wrapper integrations, context-based attribution, and OTLP-compatible delivery to Cloptima.
- Reworked the README around customer onboarding paths instead of long helper lists.
- Clarified extractor customization guidance for provider response drift.

## 0.1.0

- Initial public beta release of the Cloptima Python LLM observability SDK.
- Added `init_from_env()` for environment-based setup and disabled pass-through behavior when the SDK is not configured.
- Added `observe_call(...)`, `observe_async_call(...)`, and stream variants for instrumenting application-level LLM calls.
- Added `instrument_httpx_client(...)` and `instrument_httpx_transport(...)` for shared transport integrations.
- Added payload preview and validation helpers for local testing and CI checks.
- Added OTLP preview support and example integrations.
