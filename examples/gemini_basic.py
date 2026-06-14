import json
import urllib.request

from cloptima_llm_observability import extract_gemini_usage, init_from_env


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
        if payload.get("provider") != "gemini" or payload.get("model") != "gemini-2.5-pro":
            raise RuntimeError("unexpected telemetry payload")
        return _FakeResponse()

    urllib.request.urlopen = fake_urlopen
    try:
        client = init_from_env(
            env={
                "CLOPTIMA_LLM_OBSERVABILITY_API_KEY": "cloptima_pat_example",
                "CLOPTIMA_LLM_OBSERVABILITY_APP_ID": "support-api",
                "CLOPTIMA_LLM_OBSERVABILITY_ENVIRONMENT": "dev",
            }
        )
        client.observe_call(
            provider="gemini",
            model="gemini-2.5-pro",
            call=lambda: {
                "responseId": "gemini-example",
                "modelVersion": "gemini-2.5-pro",
                "usageMetadata": {
                    "promptTokenCount": 6,
                    "responseTokenCount": 3,
                    "totalTokenCount": 9,
                },
            },
            extract_usage=extract_gemini_usage,
            fire_and_forget=False,
            feature_id="message_classification",
        )
    finally:
        urllib.request.urlopen = original_urlopen


if __name__ == "__main__":
    main()
