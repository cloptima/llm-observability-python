import json
import urllib.request

from cloptima_llm_observability import (
    extract_openai_usage,
    init_from_env,
    wrap_observed_service,
)


class _FakeResponse:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return b"{}"


class DraftingService:
    def draft_reply(self, prompt: str):
        return {
            "id": "chatcmpl-workflow-context",
            "model": "gpt-4.1-mini",
            "input": prompt,
            "usage": {
                "prompt_tokens": 18,
                "completion_tokens": 9,
                "total_tokens": 27,
            },
        }


def main() -> None:
    original_urlopen = urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads((request.data or b"{}").decode("utf-8"))
        metadata = payload.get("metadata") or {}
        if metadata.get("workflow_id") != "support_agent" or metadata.get("feature_id") != "draft_reply":
            raise RuntimeError("workflow context was not emitted")
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
        drafting_service = wrap_observed_service(
            client,
            DraftingService(),
            {
                "draft_reply": {
                    "kind": "call",
                    "options": {
                        "provider": "openai",
                        "model": "gpt-4.1-mini",
                        "extract_usage": extract_openai_usage,
                        "fire_and_forget": False,
                    },
                }
            },
        )

        with client.with_workflow("support_agent", tenant_id="acme-prod"):
            with client.with_task("draft_reply", team_id="customer-support"):
                drafting_service.draft_reply("Draft a calm response to the customer.")
    finally:
        urllib.request.urlopen = original_urlopen


if __name__ == "__main__":
    main()
