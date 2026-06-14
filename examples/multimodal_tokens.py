import json
import urllib.request

from cloptima_llm_observability import extract_openai_usage, init_from_env

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
        if payload.get("provider") != "openai" or payload.get("model") != "gpt-4.1-mini":
            raise RuntimeError("unexpected telemetry payload")
        if payload.get("extra_usage_units") != {
            "input_audio": 24,
            "input_image": 12,
            "output_image": 8,
            "output_video": 4,
        }:
            raise RuntimeError("expected normalized multimodal token usage")
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
                "CLOPTIMA_LLM_OBSERVABILITY_TEAM_ID": "studio",
            }
        )

        client.observe_call(
            provider="openai",
            model="gpt-4.1-mini",
            call=lambda: {
                "id": "chatcmpl-multimodal-example",
                "model": "gpt-4.1-mini",
                "usage": {
                    "prompt_tokens": 240,
                    "completion_tokens": 80,
                    "total_tokens": 320,
                    "prompt_tokens_details": {
                        "audio_tokens": 24,
                        "image_tokens": 12,
                    },
                    "completion_tokens_details": {
                        "image_tokens": 8,
                        "video_tokens": 4,
                    },
                },
            },
            extract_usage=extract_openai_usage,
            fire_and_forget=False,
            feature_id="creative_generation",
            metadata={"integration_mode": "multimodal_tokens"},
        )
    finally:
        urllib.request.urlopen = original_urlopen


if __name__ == "__main__":
    main()
