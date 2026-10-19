"""API parameter fallback must preserve live streams and tool recovery state."""

import httpx2
from openai import BadRequestError
import pytest

from homeassistant.components import conversation
from homeassistant.exceptions import HomeAssistantError
from tests.test_live_streaming import text_deltas
from tests.test_tool_argument_retry import (
    TOOLS,
    _answer,
    _assert_paired_tools,
    _calls,
    _chunk,
    _results,
    _setup,
    _Stream,
    _tool,
)


def rejected(param):
    return BadRequestError(
        "private-token",
        response=httpx2.Response(
            400, request=httpx2.Request("POST", "https://example.invalid")
        ),
        body={"param": param, "code": "unsupported_parameter"},
    )


async def test_parameter_fallback_keeps_parser_feedback_success_reuse_and_shared_limit(
    hass,
):
    entity, log, create = _setup(
        hass,
        [
            _calls(_tool('{"value":1}', call_id="success")),
            _calls(_tool("{", call_id="invalid")),
            rejected("max_completion_tokens"),
            _calls(
                _tool('{"value":1}', call_id="replayed_success"),
                _tool('{"value":2}', index=1, call_id="corrected"),
            ),
            _answer(),
        ],
        options={"max_function_calls_per_conversation": 3},
    )

    await entity._async_handle_chat_log(log, TOOLS, [])

    assert create.await_count == 5
    assert entity._execute_function_tool.await_count == 2
    first_retry, accepted_retry = [
        request.kwargs for request in create.await_args_list[2:4]
    ]
    assert first_retry["messages"] == accepted_retry["messages"]
    assert first_retry["messages"][-1]["role"] == "tool"
    assert first_retry["messages"][-1]["tool_call_id"] == "invalid"
    assert "max_completion_tokens" in first_retry
    assert "max_completion_tokens" not in accepted_retry
    assert accepted_retry["max_tokens"] == first_retry["max_completion_tokens"]
    assert create.await_args.kwargs["tool_choice"] == "none"
    assert _results(log)[0].result == _results(log)[1].result
    assert len(_results(log)) == 3
    assert log.content[-1].content == "Finished."
    assert not log.unresponded_tool_results
    for request in create.await_args_list:
        _assert_paired_tools(request.kwargs["messages"])


@pytest.mark.parametrize("use_tools", [False, True])
async def test_parameter_fallback_does_not_buffer_text(hass, use_tools):
    class LiveStream(_Stream):
        async def __aiter__(self):
            for index, token in enumerate(["Live ", "text"]):
                yield _chunk(text=token)
                assert text_deltas(log) == ["Live ", "text"][: index + 1]
                self.close.assert_not_awaited()
            yield _chunk(finish="stop")

    stream = LiveStream()
    entity, log, create = _setup(hass, [rejected("max_completion_tokens"), stream])

    await entity._async_handle_chat_log(log, TOOLS if use_tools else [], [])

    assert create.await_count == 2
    entity._execute_function_tool.assert_not_awaited()
    assert log.content[-1].content == "Live text"
    assert entity._unsupported_params == {"gpt-5-mini": {"max_completion_tokens"}}
    stream.close.assert_awaited_once()


async def test_token_limit_failure_preserves_committed_results_and_next_turn(hass):
    entity, log, create = _setup(
        hass,
        [
            _calls(_tool("{}", call_id="committed")),
            rejected("max_completion_tokens"),
            rejected("max_tokens"),
        ],
    )

    with pytest.raises(HomeAssistantError, match="output token limit"):
        await entity._async_handle_chat_log(log, TOOLS, [])

    assert create.await_count == 3
    entity._execute_function_tool.assert_awaited_once()
    assert len(_results(log)) == 1
    assert log.unresponded_tool_results
    for request in create.await_args_list:
        _assert_paired_tools(request.kwargs["messages"])

    # The provider may still answer after changing to a compatible model.
    entity.subentry.data = {"chat_model": "gpt-5"}
    log.async_add_user_content(
        conversation.UserContent(content="Please explain the result.")
    )
    create.side_effect = [_answer()]
    await entity._async_handle_chat_log(log, TOOLS, [])

    entity._execute_function_tool.assert_awaited_once()
    assert log.content[-1].content == "Finished."
    assert not log.unresponded_tool_results
    _assert_paired_tools(create.await_args.kwargs["messages"])
