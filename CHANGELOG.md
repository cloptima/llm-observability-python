# Changelog

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
