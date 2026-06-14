import json
import urllib.request

from cloptima_llm_observability import create_mapped_usage_extractor, init_from_env

DEFAULT_INGEST_URL = "https://api.cloptima.ai/v1/ai/integrations/sdk/events"


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
        if request.full_url != DEFAULT_INGEST_URL:
            raise RuntimeError("unexpected telemetry endpoint")
        if payload.get("provider") != "vertex_ai" or payload.get("model") != "gemini-2.5-flash-image-preview":
            raise RuntimeError("unexpected provider mapping")
        if payload.get("provider_request_id") != "vertex-custom-1":
            raise RuntimeError("expected provider request id from mapped payload")
        if payload.get("vendor_reported_cost_usd") != 0.0842:
            raise RuntimeError("expected provider-reported cost to be preserved")
        if payload.get("extra_usage_units") != {"output_image": 96}:
            raise RuntimeError("expected mapped multimodal usage units")
        if timeout <= 0:
            raise RuntimeError("timeout must be positive")
        return _FakeResponse()

    urllib.request.urlopen = fake_urlopen

    try:
        client = init_from_env(
            env={
                "CLOPTIMA_LLM_OBSERVABILITY_API_KEY": "cloptima_pat_example",
                "CLOPTIMA_LLM_OBSERVABILITY_APP_ID": "creative-api",
                "CLOPTIMA_LLM_OBSERVABILITY_ENVIRONMENT": "dev",
            }
        )

        extract_usage = create_mapped_usage_extractor(
            defaults={
                "provider": "vertex_ai",
            },
            fields={
                "provider_request_id": "response.id",
                "model": "response.modelVersion",
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
            metadata={
                "provider_region": "response.region",
            },
        )

        client.observe_call(
            provider="vertex_ai",
            model="gemini-2.5-flash-image-preview",
            call=lambda: {
                "response": {
                    "id": "vertex-custom-1",
                    "modelVersion": "gemini-2.5-flash-image-preview",
                    "region": "us-central1",
                },
                "usage": {
                    "promptTokenCount": 1200,
                    "responseTokenCount": 96,
                    "totalTokenCount": 1296,
                    "outputImageTokenCount": 96,
                },
                "billing": {
                    "costUsd": "0.0842",
                },
            },
            extract_usage=extract_usage,
            fire_and_forget=False,
            workflow_id="creative_asset_pipeline",
            feature_id="thumbnail_generation",
        )
    finally:
        urllib.request.urlopen = original_urlopen


if __name__ == "__main__":
    main()
