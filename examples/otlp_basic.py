import json
import urllib.request

from cloptima_llm_observability import (
    extract_openai_usage,
    init_from_env,
    preview_event_payload,
    preview_otlp_request,
    validate_payload,
)


class _FakeResponse:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return b"{}"


def main() -> None:
    original_urlopen = urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads((request.data or b"{}").decode("utf-8"))
        span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        if request.full_url != "https://api.cloptima.ai/v1/ai/integrations/otlp/traces":
            raise RuntimeError("unexpected OTLP endpoint")
        if span.get("name") != "llm.openai.gpt-4.1-mini":
            raise RuntimeError("unexpected OTLP telemetry payload")
        if timeout <= 0:
            raise RuntimeError("timeout must be positive")
        return _FakeResponse()

    urllib.request.urlopen = fake_urlopen
    try:
        client = init_from_env(
            env={
                "CLOPTIMA_LLM_OBSERVABILITY_API_KEY": "cloptima_pat_example",
                "CLOPTIMA_LLM_OBSERVABILITY_APP_ID": "support-api",
                "CLOPTIMA_LLM_OBSERVABILITY_ENVIRONMENT": "dev",
                "CLOPTIMA_LLM_OBSERVABILITY_DELIVERY_MODE": "otlp_http",
                "CLOPTIMA_LLM_OBSERVABILITY_OTLP_SERVICE_NAME": "agent-api",
                "CLOPTIMA_LLM_OBSERVABILITY_OTLP_SERVICE_VERSION": "2026.06.14",
            }
        )
        preview_payload = preview_event_payload(
            {
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "app_id": "support-api",
                "environment": "dev",
                "feature_id": "customer_summary",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }
        )
        if not validate_payload(preview_payload)["valid"]:
            raise RuntimeError("preview payload should be valid")
        preview_request = preview_otlp_request(
            preview_payload,
            service_name="agent-api",
            service_version="2026.06.14",
        )
        if not preview_request["resourceSpans"][0]["resource"]["attributes"]:
            raise RuntimeError("preview request should include resource attributes")

        with client.with_workflow("support_agent", team_id="customer-support"):
            client.observe_call(
                provider="openai",
                model="gpt-4.1-mini",
                call=lambda: {
                    "id": "chatcmpl-otlp-example",
                    "model": "gpt-4.1-mini",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
                extract_usage=extract_openai_usage,
                fire_and_forget=False,
                feature_id="customer_summary",
                metadata={
                    "integration_mode": "otlp_http",
                    "deployment_shape": "enterprise",
                },
            )
    finally:
        urllib.request.urlopen = original_urlopen


if __name__ == "__main__":
    main()
