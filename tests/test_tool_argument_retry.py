"""Regression tests for bounded argument recovery using HA's real ChatLog."""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import APIConnectionError
import pytest

from custom_components.extended_openai_conversation.conversation import (
    ExtendedOpenAIAgentEntity,
)
from custom_components.extended_openai_conversation.entity import (
    ExtendedOpenAIBaseLLMEntity,
    _make_tool_result_content,
    _shorten_tool_call_id,
    _tool_result_data,
)
from custom_components.extended_openai_conversation.exceptions import (
    ParseArgumentsFailed,
    TokenLengthExceededError,
)
from homeassistant.components import conversation
from homeassistant.exceptions import HomeAssistantError

TOOLS = [
    {
        "spec": {
            "name": "do_work",
            "parameters": {"type": "object", "properties": {}},
        },
        "function": {"type": "template", "value_template": "ok"},
    }
]


def _tool(arguments, *, index=0, call_id="call_1", name="do_work", extra=None):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
        extra_content=extra,
    )


def _chunk(*calls, text=None, finish=None, usage=None):
    return SimpleNamespace(
        choices=(
            []
            if usage
            else [
                SimpleNamespace(
                    delta=SimpleNamespace(content=text, tool_calls=calls),
                    finish_reason=finish,
                )
            ]
        ),
        usage=usage,
    )


class _Stream:
    """An SDK-like stream with observable cleanup and injected transport errors."""

    def __init__(self, *chunks):
        self.chunks = chunks
        self.close = AsyncMock()

    async def __aiter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


def _calls(*calls, finish="tool_calls"):
    return _Stream(_chunk(*calls, finish=finish))


def _answer():
    return _Stream(_chunk(text="Finished.", finish="stop"))


def _setup(hass, streams, *, options=None, agent_class=ExtendedOpenAIBaseLLMEntity):
    entry = MagicMock()
    entry.runtime_data.chat.completions.create = AsyncMock(side_effect=streams)
    subentry = MagicMock(subentry_id="test", title="Test agent")
    subentry.data = options or {}
    entity = agent_class(entry, subentry)
    entity.hass = hass
    entity.entity_id = "conversation.test"
    chat_log = conversation.ChatLog(hass, "test_conversation")
    chat_log.async_add_user_content(conversation.UserContent(content="Do the work"))
    chat_log.async_trace = MagicMock()
    chat_log.delta_listener = MagicMock()

    async def execute(function_tool, tool_input, llm_context, exposed_entities):
        return _make_tool_result_content(
            entity.entity_id,
            tool_input.id,
            tool_input.tool_name,
            {"result": str(tool_input.tool_args)},
        )

    entity._execute_function_tool = AsyncMock(side_effect=execute)
    return entity, chat_log, entry.runtime_data.chat.completions.create


async def _run(entity, chat_log, tools=TOOLS):
    await entity._async_handle_chat_log(chat_log, tools, [])


def _results(chat_log):
    return [c for c in chat_log.content if c.role == "tool_result"]


@pytest.mark.parametrize("use_tools", [False, True])
async def test_normal_requests_keep_existing_flow(hass, use_tools):
    streams = [_calls(_tool('{"value":1}')), _answer()] if use_tools else [_answer()]
    entity, log, create = _setup(hass, streams)

    await _run(entity, log)

    assert create.await_count == len(streams)
    assert entity._execute_function_tool.await_count == int(use_tools)
    assert log.content[-1].content == "Finished."
    assert not log.unresponded_tool_results
    assert all(
        len(call.kwargs["messages"]) == 2
        or call.kwargs["messages"][-1]["role"] == "tool"
        for call in create.await_args_list
    )
    for stream in streams:
        stream.close.assert_awaited_once()


@pytest.mark.parametrize("finish", ["tool_calls", "stop"])
async def test_interleaved_argument_fragments_are_not_parsed_early(hass, finish):
    stream = _Stream(
        _chunk(_tool('{"a":', call_id="one"), _tool('{"b":', index=1, call_id="two")),
        _chunk(
            _tool("1}", call_id=None, name=None),
            _tool("2}", index=1, call_id=None, name=None),
            finish=finish,
        ),
    )
    entity, log, create = _setup(hass, [stream, _answer()])

    await _run(entity, log)

    assert create.await_count == 2
    assert [
        c.args[1].tool_args for c in entity._execute_function_tool.await_args_list
    ] == [
        {"a": 1},
        {"b": 2},
    ]


@pytest.mark.parametrize("arguments", ['{"value":', "", "null", "[]", '"text"'])
async def test_invalid_arguments_are_regenerated_once(hass, arguments):
    failed = _calls(_tool(arguments))
    repaired = _calls(_tool('{"value":1}', call_id="repaired"))
    entity, log, create = _setup(hass, [failed, repaired, _answer()])

    await _run(entity, log)

    assert create.await_count == 3
    entity._execute_function_tool.assert_awaited_once()
    assert entity._execute_function_tool.await_args.args[1].id == "repaired"
    retry_messages = create.await_args_list[1].kwargs["messages"]
    assert retry_messages[:-1] == create.await_args_list[0].kwargs["messages"]
    assert retry_messages[-1]["role"] == "system"
    assert [c.role for c in log.content] == [
        "system",
        "user",
        "assistant",
        "tool_result",
        "assistant",
    ]
    assert log.content[2].tool_calls[0].id == "repaired"
    assert "Regenerate" not in str([c.as_dict() for c in log.content])
    failed.close.assert_awaited_once()


@pytest.mark.parametrize("separate_batches", [False, True])
async def test_failed_stream_executes_no_partial_batch(hass, separate_batches):
    good = _tool('{"value":1}', call_id="old_one")
    bad = _tool('{"value":', index=1, call_id="old_two")
    failed = (
        _Stream(_chunk(good, finish="tool_calls"), _chunk(bad, finish="tool_calls"))
        if separate_batches
        else _calls(good, bad)
    )
    repaired = _calls(
        _tool('{"value":1}', call_id="new_one"),
        _tool('{"value":2}', index=1, call_id="new_two"),
    )
    entity, log, create = _setup(hass, [failed, repaired, _answer()])
    responses = iter([failed, repaired, _answer()])

    async def respond(**kwargs):
        if create.await_count == 2:
            entity._execute_function_tool.assert_not_awaited()
            failed.close.assert_awaited_once()
        return next(responses)

    create.side_effect = respond
    await _run(entity, log)

    assert [c.args[1].id for c in entity._execute_function_tool.await_args_list] == [
        "new_one",
        "new_two",
    ]
    assert len(_results(log)) == 2


@pytest.mark.parametrize("shorten", [False, True])
async def test_correction_reuses_prior_success_even_with_new_id(hass, shorten):
    signature = {"google": {"thought_signature": "retry-signature"}}
    streams = [
        _calls(_tool('{"a":1,"b":2}', call_id="completed")),
        _calls(_tool('{"a":', call_id="broken")),
        _calls(
            _tool('{"b":2,"a":1}', call_id="replay", extra=signature),
            _tool('{"a":3}', index=1, call_id="new"),
        ),
        _answer(),
    ]
    entity, log, create = _setup(
        hass, streams, options={"shorten_tool_call_id": shorten}
    )

    await _run(entity, log)

    assert [c.args[1].id for c in entity._execute_function_tool.await_args_list] == [
        "completed",
        "new",
    ]
    results = _results(log)
    assert [r.tool_call_id for r in results] == ["completed", "replay", "new"]
    assert _tool_result_data(results[0]) == _tool_result_data(results[1])
    messages = create.await_args_list[-1].kwargs["messages"]
    assistant = messages[-3]
    expected_id = _shorten_tool_call_id("replay") if shorten else "replay"
    assert assistant["tool_calls"][0]["id"] == expected_id
    assert assistant["tool_calls"][0]["extra_content"] == signature
    assert messages[-2]["tool_call_id"] == expected_id
    assert not log.unresponded_tool_results


async def test_normal_intentional_repeat_is_not_deduplicated(hass):
    entity, log, _ = _setup(
        hass,
        [
            _calls(_tool("{}", call_id="one")),
            _calls(_tool("{}", call_id="two")),
            _answer(),
        ],
    )

    await _run(entity, log)

    assert entity._execute_function_tool.await_count == 2


async def test_correction_does_not_schedule_completed_background_tool_again(
    hass, monkeypatch
):
    delayed = '{"value":1,"delay":{"seconds":1}}'
    entity, log, _ = _setup(
        hass,
        [
            _calls(_tool(delayed, call_id="scheduled")),
            _calls(_tool("{")),
            _calls(
                _tool(delayed, call_id="replay"),
                _tool('{"value":2}', index=1, call_id="new"),
            ),
            _answer(),
        ],
    )
    del entity._execute_function_tool
    function = MagicMock(execute=AsyncMock(return_value="ok"))
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation.entity.get_function",
        MagicMock(return_value=function),
    )
    entity.entry.async_create_task.side_effect = lambda hass, coro: coro.close()

    await _run(entity, log)

    entity.entry.async_create_task.assert_called_once()
    assert function.execute.call_count == 2
    assert function.execute.await_count == 1
    assert _tool_result_data(_results(log)[1]) == {"result": "Scheduled"}


async def test_retry_budget_is_shared_across_iterations(hass):
    streams = [_calls(_tool("{")), _calls(_tool("{}")), _calls(_tool("{"))]
    entity, log, create = _setup(hass, streams)

    with pytest.raises(ParseArgumentsFailed):
        await _run(entity, log)

    assert create.await_count == 3
    entity._execute_function_tool.assert_awaited_once()
    assert len(_results(log)) == 1
    assert log.content[-1].role == "tool_result"
    for stream in streams:
        stream.close.assert_awaited_once()


async def test_two_failures_restore_request_checkpoint(hass):
    entity, log, create = _setup(hass, [_calls(_tool("{")), _calls(_tool("{"))])
    checkpoint = log.content.copy()

    with pytest.raises(ParseArgumentsFailed):
        await _run(entity, log)

    assert create.await_count == 2
    assert log.content == checkpoint
    entity._execute_function_tool.assert_not_awaited()


async def test_retry_budget_and_replay_cache_reset_for_next_user_turn(hass):
    entity, log, create = _setup(
        hass,
        [
            _calls(_tool("{")),
            _calls(_tool("{}", call_id="first_turn")),
            _answer(),
            _calls(_tool("{")),
            _calls(_tool("{}", call_id="second_turn")),
            _answer(),
        ],
    )

    await _run(entity, log)
    log.async_add_user_content(conversation.UserContent(content="Do it again"))
    await _run(entity, log)

    assert create.await_count == 6
    assert entity._execute_function_tool.await_count == 2
    assert sum(c.role == "user" for c in log.content) == 2
    assert not log.unresponded_tool_results


@pytest.mark.parametrize("arguments", ['{"value":', '{"value":1}'])
async def test_eof_with_pending_calls_does_not_silently_drop_them(hass, arguments):
    incomplete = _calls(_tool(arguments), finish=None)
    entity, log, create = _setup(
        hass, [incomplete, _calls(_tool('{"value":1}')), _answer()]
    )

    await _run(entity, log)

    assert create.await_count == 3
    entity._execute_function_tool.assert_awaited_once()
    incomplete.close.assert_awaited_once()


@pytest.mark.parametrize("failure", ["length", "network", "cancel", "filtered"])
async def test_other_stream_failures_are_not_argument_retried(hass, failure):
    if failure == "network":
        stream = _Stream(
            APIConnectionError(request=httpx.Request("POST", "https://example.test"))
        )
        expected = APIConnectionError
    elif failure == "cancel":
        stream = _Stream(asyncio.CancelledError())
        expected = asyncio.CancelledError
    else:
        stream = _calls(
            _tool("{"), finish="length" if failure == "length" else "content_filter"
        )
        expected = (
            TokenLengthExceededError if failure == "length" else HomeAssistantError
        )
    entity, log, create = _setup(hass, [stream])

    with pytest.raises(expected):
        await _run(entity, log)

    assert create.await_count == 1
    entity._execute_function_tool.assert_not_awaited()
    stream.close.assert_awaited_once()


@pytest.mark.parametrize("after_correction", [False, True])
async def test_executor_errors_never_restart_generation(hass, after_correction):
    streams = ([_calls(_tool("{"))] if after_correction else []) + [_calls(_tool("{}"))]
    entity, log, create = _setup(hass, streams)
    entity._execute_function_tool.side_effect = ParseArgumentsFailed(
        "private executor data"
    )

    with pytest.raises(ParseArgumentsFailed):
        await _run(entity, log)

    assert create.await_count == len(streams)
    entity._execute_function_tool.assert_awaited_once()


@pytest.mark.parametrize(
    "invalid_repair", ["text_only", "duplicate_id", "changed_id_arguments"]
)
async def test_invalid_correction_is_not_executed(hass, invalid_repair):
    repair = {
        "text_only": _answer(),
        "duplicate_id": _calls(_tool("{}"), _tool("{}", index=1)),
        "changed_id_arguments": _calls(_tool('{"value":2}', call_id="completed")),
    }[invalid_repair]
    entity, log, create = _setup(
        hass,
        [_calls(_tool('{"value":1}', call_id="completed")), _calls(_tool("{")), repair],
    )

    with pytest.raises(ParseArgumentsFailed):
        await _run(entity, log)

    assert create.await_count == 3
    entity._execute_function_tool.assert_awaited_once()
    assert [r.tool_call_id for r in _results(log)] == ["completed"]


async def test_retry_preserves_limits_and_request_options(hass):
    entity, log, create = _setup(
        hass,
        [_calls(_tool("{")), _calls(_tool("{}")), _answer()],
        options={"max_function_calls_per_conversation": 1, "temperature": 0.3},
    )

    await _run(entity, log)

    requests = [c.kwargs for c in create.await_args_list]
    assert [{k: v for k, v in r.items() if k != "messages"} for r in requests[:2]] == [
        {k: v for k, v in requests[0].items() if k != "messages"}
    ] * 2
    assert [r["tool_choice"] for r in requests] == ["auto", "auto", "none"]


async def test_no_configured_tools_means_no_correction_request(hass):
    entity, log, create = _setup(hass, [_calls(_tool("{"))])

    with pytest.raises(ParseArgumentsFailed):
        await _run(entity, log, tools=[])

    assert create.await_count == 1


async def test_streamed_text_and_native_are_not_replayed_during_correction(hass):
    signature = {"google": {"thought_signature": "accepted-signature"}}
    failed = _Stream(_chunk(text="Working."), _chunk(_tool("{"), finish="tool_calls"))
    repaired = _Stream(
        _chunk(text="Working."),
        _chunk(_tool("{}", extra=signature), finish="tool_calls"),
    )
    entity, log, _ = _setup(hass, [failed, repaired, _answer()])

    await _run(entity, log)

    deltas = [c.args[1] for c in log.delta_listener.call_args_list]
    assert [d["content"] for d in deltas if "content" in d] == ["Working.", "Finished."]
    assert sum(d.get("role") == "assistant" for d in deltas) == 2
    assert log.content[2].content is None
    assert log.content[2].native == {
        "tool_call_extra_content_by_id": {"call_1": signature}
    }


async def test_failed_attempt_truncation_does_not_remove_successful_history(hass):
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
    failed = _Stream(_chunk(usage=usage), _chunk(_tool("{"), finish="tool_calls"))
    entity, log, create = _setup(
        hass,
        [
            _calls(_tool("{}", call_id="completed")),
            failed,
            _calls(_tool('{"value":2}')),
            _answer(),
        ],
        options={"context_threshold": 50, "context_truncate_strategy": "clear"},
    )
    log.content.insert(1, conversation.UserContent(content="Previous turn"))
    log.content.insert(
        2,
        conversation.AssistantContent(
            agent_id=entity.entity_id, content="Earlier answer"
        ),
    )

    await _run(entity, log)

    assert log.content[1].content == "Previous turn"
    assert len(_results(log)) == 2
    assert (
        create.await_args_list[2].kwargs["messages"][:-1]
        == create.await_args_list[1].kwargs["messages"]
    )
    log.async_trace.assert_called_once_with(
        {"stats": {"input_tokens": 100, "output_tokens": 20}}
    )


@pytest.mark.parametrize("new_conversation", [False, True])
async def test_terminal_error_and_diagnostics_are_redacted(
    hass, caplog, new_conversation
):
    secret = "private-token-device-command"
    broken = json.dumps({"secret": secret})[:-1]
    entity, log, create = _setup(
        hass,
        [
            _calls(_tool("{}", call_id="completed")),
            _calls(_tool(broken)),
            _calls(_tool(broken)),
        ],
        agent_class=ExtendedOpenAIAgentEntity,
    )
    entity._get_exposed_entities = MagicMock(return_value=[])
    entity._get_function_tools = MagicMock(return_value=TOOLS)
    entity._build_system_prompt = MagicMock(return_value="System prompt")
    user_input = MagicMock(
        language="en", conversation_id=None if new_conversation else log.conversation_id
    )
    user_input.as_llm_context.return_value = None
    caplog.set_level(
        logging.DEBUG, logger="custom_components.extended_openai_conversation"
    )

    result = await entity._async_handle_message(user_input, log)

    assert create.await_count == 3
    assert result.response.error_code is not None
    assert result.conversation_id == log.conversation_id
    assert log.content[-1].role == "assistant"
    assert "Please try again" in log.content[-1].content
    assert not log.unresponded_tool_results
    assert not log.continue_conversation
    assert secret not in str(result.as_dict())
    assert secret not in caplog.text
    assert "arguments_length=" in caplog.text
    assert "retry_used=True" in caplog.text
    error = ParseArgumentsFailed(broken)
    assert secret not in str(error)
    assert secret not in repr(error.args)
    assert secret not in repr(vars(error))
