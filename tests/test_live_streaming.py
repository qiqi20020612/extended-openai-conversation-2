"""Text must reach HA before the provider emits the next chunk or finishes."""

import pytest

from homeassistant.exceptions import HomeAssistantError
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


def text_deltas(log):
    return [
        call.args[1]["content"]
        for call in log.delta_listener.call_args_list
        if "content" in call.args[1]
    ]


@pytest.mark.parametrize(
    "scenario",
    [
        "no_tools",
        "configured_tools",
        "argument_error",
        "execution_error",
        "after_tool",
        "exhausted_budget",
    ],
)
async def test_text_is_forwarded_before_next_chunk_and_response_end(hass, scenario):
    class LiveStream(_Stream):
        async def __aiter__(self):
            tokens = ["Still ", "streaming", "."]
            for index, token in enumerate(tokens):
                yield _chunk(text=token)
                # Runs before the provider supplies another chunk or finish_reason.
                assert text_deltas(log) == tokens[: index + 1]
                self.close.assert_not_awaited()
            yield _chunk(finish="stop")

    final = LiveStream()
    before = []
    if scenario == "argument_error":
        before = [_calls(_tool("{"))]
    elif scenario in {"execution_error", "after_tool", "exhausted_budget"}:
        before = [_calls(_tool("{}"))]
    entity, log, create = _setup(
        hass,
        [*before, final],
        options={
            "max_function_calls_per_conversation": 1
            if scenario == "exhausted_budget"
            else 2
        },
    )
    if scenario == "execution_error":
        entity._execute_function_tool.side_effect = HomeAssistantError("Tool failed")

    await entity._async_handle_chat_log(
        log, [] if scenario == "no_tools" else TOOLS, []
    )

    assert text_deltas(log) == ["Still ", "streaming", "."]
    assert log.content[-1].content == "Still streaming."
    assert create.await_count == len(before) + 1
    assert entity._execute_function_tool.await_count == int(
        scenario in {"execution_error", "after_tool", "exhausted_budget"}
    )
    if scenario == "execution_error":
        assert _results(log)[0].result.error
    if scenario == "exhausted_budget":
        assert create.await_args.kwargs["tool_choice"] == "none"
    final.close.assert_awaited_once()
    _assert_paired_tools(create.await_args.kwargs["messages"])
    assert not log.unresponded_tool_results


@pytest.mark.parametrize("after_argument_error", [False, True])
async def test_interleaved_tool_arguments_do_not_delay_text_or_execute_early(
    hass, after_argument_error
):
    class ToolStream(_Stream):
        async def __aiter__(self):
            yield _chunk(_tool('{"value":', call_id="complete_later"), text="Working ")
            assert text_deltas(log) == ["Working "]
            entity._execute_function_tool.assert_not_awaited()
            yield _chunk(_tool("1}", call_id=None, name=None), text="now.")
            assert text_deltas(log) == ["Working ", "now."]
            entity._execute_function_tool.assert_not_awaited()
            # Complete JSON is still not executed before the response finishes.
            yield _chunk(finish="tool_calls")
            entity._execute_function_tool.assert_not_awaited()
            self.close.assert_not_awaited()

    class FinalStream(_Stream):
        async def __aiter__(self):
            entity._execute_function_tool.assert_awaited_once()
            first.close.assert_awaited_once()
            yield _chunk(text="All ")
            assert text_deltas(log) == ["Working ", "now.", "All "]
            yield _chunk(text="done.")
            assert text_deltas(log) == ["Working ", "now.", "All ", "done."]
            yield _chunk(finish="stop")

    first, final = ToolStream(), FinalStream()
    before = [_calls(_tool("{"))] if after_argument_error else []
    entity, log, create = _setup(
        hass,
        [*before, first, final],
        options={"max_function_calls_per_conversation": 2},
    )

    await entity._async_handle_chat_log(log, TOOLS, [])

    assert entity._execute_function_tool.await_args.args[1].tool_args == {"value": 1}
    assert log.content[2].content == "Working now."
    assert log.content[-1].content == "All done."
    assert len(_results(log)) == 1
    first.close.assert_awaited_once()
    final.close.assert_awaited_once()
    for request in create.await_args_list:
        _assert_paired_tools(request.kwargs["messages"])


async def test_conflicting_tool_id_retries_clean_history_without_reexecuting(hass):
    entity, log, create = _setup(
        hass,
        [
            _calls(_tool('{"value":1}', call_id="completed")),
            _Stream(
                _chunk(text="Another operation."),
                _chunk(_tool('{"value":2}', call_id="completed"), finish="tool_calls"),
            ),
            _answer(),
        ],
        options={"max_function_calls_per_conversation": 3},
    )

    await entity._async_handle_chat_log(log, TOOLS, [])

    entity._execute_function_tool.assert_awaited_once()
    messages = create.await_args.kwargs["messages"]
    assert messages == create.await_args_list[1].kwargs["messages"]
    assert [result.tool_call_id for result in _results(log)] == ["completed"]
    assert text_deltas(log) == ["Another operation.", "Finished."]
    assert log.content[-1].content == "Finished."
    for request in create.await_args_list:
        _assert_paired_tools(request.kwargs["messages"])
