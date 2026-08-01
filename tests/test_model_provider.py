from __future__ import annotations

import json

import httpx
import pytest

from deep_researcher.kernel import ModelRequest
from deep_researcher.providers.model import (
    OpenAICompatibleModelAdapter,
    OpenAICompatibleModelConfig,
)


def _request() -> ModelRequest:
    return ModelRequest(
        run_id="run_model_provider",
        task_id="task_model_provider",
        actor_id="agent_model_provider",
        system="Return bounded structured commands.",
        messages=(
            {
                "role": "user",
                "content": {
                    "message_type": "governed_tool_observation",
                    "observation": {"items": ["evidence"]},
                },
            },
        ),
        command_schema={
            "type": "object",
            "required": ["commands"],
            "properties": {"commands": {"type": "array"}},
        },
        model_version="deepseek-chat@test",
        prompt_version="worker@test",
        max_output_tokens=512,
    )


@pytest.mark.asyncio
async def test_malformed_json_is_preserved_for_bounded_kernel_repair():
    requests: list[dict] = []
    malformed = '{"commands":[{"kind":"stop" "name":"stop"}]}'
    corrected = {
        "commands": [
            {"kind": "stop", "name": "stop", "arguments": {}}
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        content = malformed if len(requests) == 1 else json.dumps(corrected)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": f"response-{len(requests)}",
                "choices": [
                    {
                        "message": {"content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8},
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        adapter = OpenAICompatibleModelAdapter(
            OpenAICompatibleModelConfig(api_key="test-key"),
            http_client=client,
        )
        invalid = await adapter.complete(_request())
        repaired = await adapter.repair(
            _request(),
            invalid,
            ("commands[0] is invalid JSON",),
        )

    assert invalid.content == malformed
    assert invalid.structured is None
    assert repaired.structured == corrected
    assert all(
        message["role"] != "tool"
        for request in requests
        for message in request["messages"]
    )
    repair_payload = json.loads(requests[1]["messages"][-1]["content"])
    assert repair_payload["invalid_response"] == malformed
    assert repair_payload["validation_errors"] == [
        "commands[0] is invalid JSON"
    ]
