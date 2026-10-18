"""Regression tests for bounded argument recovery using HA's real ChatLog."""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import APIConnectionError, AsyncOpenAI, BadRequestError
import probatio as vol
import pytest

from custom_components.extended_openai_conversation.conversation import (
    ExtendedOpenAIAgentEntity,
)
from custom_components.extended_openai_conversation.entity import (
    MAX_TOOL_ITERATIONS,
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


def _assert_paired_tools(messages):
    """No orphan, missing, duplicate or interrupted tool-result pairs on the wire."""
    pending = set()
    for message in messages:
        if message["role"] == "tool":
            assert message["tool_call_id"] in pending
            pending.remove(message["tool_call_id"])
        else:
            assert not pending
            calls = message.get("tool_calls", [])
            pending = {call["id"] for call in calls}
            assert len(pending) == len(calls)
    assert not pending


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
    entity, log, create = _setup(
        hass,
        [stream, _answer()],
        options={"max_function_calls_per_conversation": 1},
    )

    await _run(entity, log)

    assert create.await_count == 2
    assert [
        c.args[1].tool_args for c in entity._execute_function_tool.await_args_list
    ] == [
        {"a": 1},
        {"b": 2},
    ]
    assert [c.kwargs["tool_choice"] for c in create.await_args_list] == ["auto", "none"]


@pytest.mark.parametrize("arguments", ['{"value":', "", "null", "[]", '"text"'])
async def test_invalid_arguments_use_parser_feedback(hass, arguments):
    failed = _calls(_tool(arguments))
    repaired = _calls(_tool('{"value":1}', call_id="repaired"))
    entity, log, create = _setup(
        hass,
        [failed, repaired, _answer()],
        options={"max_function_calls_per_conversation": 2},
    )

    await _run(entity, log)

    assert create.await_count == 3
    entity._execute_function_tool.assert_awaited_once()
    assert entity._execute_function_tool.await_args.args[1].id == "repaired"
    original_messages = create.await_args_list[0].kwargs["messages"]
    retry_messages = create.await_args_list[1].kwargs["messages"]
    assert retry_messages[: len(original_messages)] == original_messages
    feedback = retry_messages[len(original_messages) :]
    assert [m["role"] for m in feedback] == ["assistant", "tool"]
    assert feedback[0]["tool_calls"][0]["function"]["arguments"] == arguments
    try:
        vol.Schema(dict)(json.loads(arguments))
    except (json.JSONDecodeError, vol.Invalid) as error:
        assert json.loads(feedback[1]["content"]) == {
            "error": type(error).__name__,
            "error_text": str(error),
        }
    else:
        pytest.fail("Expected an actual parser or validator error")
    _assert_paired_tools(retry_messages)
    assert sum(m["role"] == "system" for m in retry_messages) == 1
    assert [c.role for c in log.content] == [
        "system",
        "user",
        "assistant",
        "tool_result",
        "assistant",
    ]
    assert log.content[2].tool_calls[0].id == "repaired"
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
    entity, log, create = _setup(
        hass,
        [failed, repaired, _answer()],
        options={"max_function_calls_per_conversation": 2},
    )
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
    assert [c.kwargs["tool_choice"] for c in create.await_args_list] == [
        "auto",
        "auto",
        "none",
    ]
    feedback = create.await_args_list[1].kwargs["messages"][2:]
    assert [m["role"] for m in feedback] == ["assistant", "tool", "tool"]
    assert json.loads(feedback[1]["content"]) == {"status": "not_executed"}
    assert json.loads(feedback[2]["content"])["error"] == "JSONDecodeError"
    _assert_paired_tools(create.await_args_list[1].kwargs["messages"])


@pytest.mark.parametrize("shorten", [False, True])
@pytest.mark.parametrize("missing_identity", [False, True])
async def test_correction_reuses_prior_success_even_with_new_id(
    hass, shorten, missing_identity
):
    signature = {"google": {"thought_signature": "retry-signature"}}
    failed_signature = {"google": {"thought_signature": "failed-call-signature"}}
    streams = [
        _calls(_tool('{"a":1,"b":2}', call_id="completed")),
        _calls(
            _tool(
                '{"a":',
                call_id=None if missing_identity else "broken",
                extra=failed_signature,
            )
        ),
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
    if missing_identity:
        assert (
            create.await_args_list[2].kwargs["messages"]
            == create.await_args_list[1].kwargs["messages"]
        )
    else:
        feedback = create.await_args_list[2].kwargs["messages"][-2:]
        assert feedback[0]["tool_calls"][0]["extra_content"] == failed_signature
        broken_id = _shorten_tool_call_id("broken") if shorten else "broken"
        assert feedback[1]["tool_call_id"] == broken_id
    _assert_paired_tools(create.await_args_list[2].kwargs["messages"])


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
    assert function.execute.call_count == 1
    assert function.execute.await_count == 1
    assert _tool_result_data(_results(log)[1])["result"] == "Scheduled"
    assert _tool_result_data(_results(log)[1])["execution_status"] == "scheduled"


@pytest.mark.parametrize("limit", [1, 2, 3, 10, -1])
@pytest.mark.parametrize("failure", ["json", "validator", "missing_id", "eof"])
async def test_corrections_share_tool_round_limit_and_allow_final_text(
    hass, limit, failure
):
    rounds = limit if limit >= 0 else MAX_TOOL_ITERATIONS
    failures = [
        _calls(
            _tool(
                "[]" if failure == "validator" else "{",
                call_id=None if failure == "missing_id" else f"failed_{index}",
            ),
            finish=None if failure == "eof" else "tool_calls",
        )
        for index in range(rounds - 1)
    ]
    streams = [*failures, _calls(_tool("{}", call_id="valid")), _answer()]
    entity, log, create = _setup(
        hass, streams, options={"max_function_calls_per_conversation": limit}
    )

    await _run(entity, log)

    assert create.await_count == rounds + 1
    entity._execute_function_tool.assert_awaited_once()
    assert entity._execute_function_tool.await_args.args[1].id == "valid"
    assert [c.kwargs["tool_choice"] for c in create.await_args_list] == [
        *(["auto"] * rounds),
        "none",
    ]
    original = create.await_args_list[0].kwargs["messages"]
    for index, request in enumerate(create.await_args_list[1:rounds], start=1):
        messages = request.kwargs["messages"]
        _assert_paired_tools(messages)
        if failure in {"missing_id", "eof"}:
            assert messages == original
        else:
            assert messages[: len(original)] == original
            assert [m["role"] for m in messages[len(original) :]] == [
                "assistant",
                "tool",
            ]
            assert messages[-1]["tool_call_id"] == f"failed_{index - 1}"
    assert [r.tool_call_id for r in _results(log)] == ["valid"]
    assert log.content[-1].content == "Finished."
    assert not log.unresponded_tool_results
    for stream in streams:
        stream.close.assert_awaited_once()


@pytest.mark.parametrize("limit", [1, 2, 3, 10, -1, MAX_TOOL_ITERATIONS + 10])
@pytest.mark.parametrize("failure", ["json", "validator", "missing_id", "eof"])
async def test_exhausted_tool_round_limit_stops_invalid_arguments(hass, limit, failure):
    rounds = min(limit, MAX_TOOL_ITERATIONS) if limit >= 0 else MAX_TOOL_ITERATIONS
    streams = [
        _calls(
            _tool(
                "[]" if failure == "validator" else "{",
                call_id=None if failure == "missing_id" else f"failed_{index}",
            ),
            finish=None if failure == "eof" else "tool_calls",
        )
        for index in range(rounds)
    ]
    entity, log, create = _setup(
        hass, streams, options={"max_function_calls_per_conversation": limit}
    )
    checkpoint = log.content.copy()

    with pytest.raises(ParseArgumentsFailed):
        await _run(entity, log)

    assert create.await_count == rounds
    assert log.content == checkpoint
    entity._execute_function_tool.assert_not_awaited()
    for request, stream in zip(create.await_args_list, streams, strict=True):
        assert request.kwargs["tool_choice"] == "auto"
        _assert_paired_tools(request.kwargs["messages"])
        stream.close.assert_awaited_once()


@pytest.mark.parametrize("limit", [1, 2, 3, 10, -1])
async def test_alternating_execution_and_correction_share_counter(hass, limit):
    rounds = limit if limit >= 0 else MAX_TOOL_ITERATIONS
    streams = []
    for index in range(rounds):
        if index % 2:
            streams.append(_calls(_tool("{", call_id=f"failed_{index}")))
        else:
            calls = (
                [
                    _tool(
                        json.dumps({"value": index // 2 - 1}),
                        call_id=f"replayed_{index}",
                    )
                ]
                if index
                else []
            )
            calls.append(
                _tool(
                    json.dumps({"value": index // 2}),
                    index=len(calls),
                    call_id=f"success_{index}",
                )
            )
            streams.append(_calls(*calls))
    if rounds % 2:
        streams.append(_answer())
    entity, log, create = _setup(
        hass, streams, options={"max_function_calls_per_conversation": limit}
    )

    if rounds % 2:
        await _run(entity, log)
        assert log.content[-1].content == "Finished."
        assert not log.unresponded_tool_results
    else:
        with pytest.raises(ParseArgumentsFailed):
            await _run(entity, log)
        assert log.content[-1].role == "tool_result"

    assert create.await_count == len(streams)
    assert [
        c.args[1].tool_args for c in entity._execute_function_tool.await_args_list
    ] == [{"value": index} for index in range((rounds + 1) // 2)]
    assert all(
        tool_input.tool_args
        for content in log.content
        if content.role == "assistant"
        for tool_input in content.tool_calls or []
    )
    for request, stream in zip(create.await_args_list, streams, strict=True):
        _assert_paired_tools(request.kwargs["messages"])
        stream.close.assert_awaited_once()


async def test_repeated_corrections_reuse_successes_but_normal_repeats_execute(hass):
    entity, log, create = _setup(
        hass,
        [
            _calls(_tool('{"value":1}', call_id="first")),
            _calls(_tool("{", call_id="bad_json")),
            _calls(_tool("[]", call_id="bad_type")),
            _calls(
                _tool('{"value":1}', call_id="replay_first"),
                _tool('{"value":2}', index=1, call_id="second"),
            ),
            _calls(_tool('{"value":2}', call_id="intentional_repeat")),
            _calls(_tool("{", call_id="bad_json_again")),
            _calls(_tool("[]", call_id="bad_type_again")),
            _calls(
                _tool('{"value":2}', call_id="replay_second"),
                _tool('{"value":3}', index=1, call_id="third"),
            ),
            _calls(_tool('{"value":4}', call_id="fourth")),
            _calls(_tool('{"value":5}', call_id="fifth")),
            _answer(),
        ],
        options={"max_function_calls_per_conversation": 10},
    )

    await _run(entity, log)

    assert create.await_count == 11
    assert [c.args[1].id for c in entity._execute_function_tool.await_args_list] == [
        "first",
        "second",
        "intentional_repeat",
        "third",
        "fourth",
        "fifth",
    ]
    results = _results(log)
    assert _tool_result_data(results[0]) == _tool_result_data(results[1])
    assert _tool_result_data(results[3]) == _tool_result_data(results[4])
    for request in create.await_args_list:
        _assert_paired_tools(request.kwargs["messages"])
    assert create.await_args.kwargs["tool_choice"] == "none"
    assert log.content[-1].content == "Finished."
    assert not log.unresponded_tool_results


@pytest.mark.parametrize("limit", [1, 2, 3, 10, -1, MAX_TOOL_ITERATIONS + 10])
async def test_normal_tools_also_reach_limit_and_finish_with_text(hass, limit):
    rounds = min(limit, MAX_TOOL_ITERATIONS) if limit >= 0 else MAX_TOOL_ITERATIONS
    streams = [
        _calls(_tool(json.dumps({"value": index}), call_id=f"call_{index}"))
        for index in range(rounds)
    ] + [_answer()]
    entity, log, create = _setup(
        hass, streams, options={"max_function_calls_per_conversation": limit}
    )

    await _run(entity, log)

    assert create.await_count == rounds + 1
    assert entity._execute_function_tool.await_count == rounds
    assert [c.kwargs["tool_choice"] for c in create.await_args_list] == [
        *(["auto"] * rounds),
        "none",
    ]
    assert log.content[-1].content == "Finished."
    assert not log.unresponded_tool_results


@pytest.mark.parametrize("limit", [0, 1])
@pytest.mark.parametrize("arguments", ["{}", "{"])
async def test_backend_cannot_execute_tools_after_limit(hass, limit, arguments):
    streams = ([_calls(_tool('{"value":1}', call_id="completed"))] if limit else []) + [
        _calls(_tool(arguments, call_id="over_limit"))
    ]
    entity, log, create = _setup(
        hass, streams, options={"max_function_calls_per_conversation": limit}
    )

    with pytest.raises(HomeAssistantError):
        await _run(entity, log)

    assert create.await_count == limit + 1
    assert create.await_args.kwargs["tool_choice"] == "none"
    assert entity._execute_function_tool.await_count == limit
    assert [r.tool_call_id for r in _results(log)] == (["completed"] if limit else [])
    assert log.content[-1].role == ("tool_result" if limit else "user")
    assert "over_limit" not in str(log.as_dict())
    for stream in streams:
        stream.close.assert_awaited_once()


async def test_zero_tool_budget_still_allows_text(hass):
    entity, log, create = _setup(
        hass, [_answer()], options={"max_function_calls_per_conversation": 0}
    )

    await _run(entity, log)

    create.assert_awaited_once()
    assert create.await_args.kwargs["tool_choice"] == "none"
    entity._execute_function_tool.assert_not_awaited()
    assert log.content[-1].content == "Finished."


@pytest.mark.parametrize("first_missing_id", [False, True])
@pytest.mark.parametrize("second_missing_id", [False, True])
async def test_two_failures_restore_request_checkpoint(
    hass, first_missing_id, second_missing_id
):
    entity, log, create = _setup(
        hass,
        [
            _calls(_tool("{", call_id=None if first_missing_id else "first")),
            _calls(_tool("{", call_id=None if second_missing_id else "second")),
        ],
        options={"max_function_calls_per_conversation": 2},
    )
    checkpoint = log.content.copy()

    with pytest.raises(ParseArgumentsFailed):
        await _run(entity, log)

    assert create.await_count == 2
    assert log.content == checkpoint
    entity._execute_function_tool.assert_not_awaited()
    _assert_paired_tools(create.await_args_list[1].kwargs["messages"])


async def test_shared_tool_budget_and_replay_cache_reset_for_next_user_turn(hass):
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
        options={"max_function_calls_per_conversation": 2},
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
    assert (
        create.await_args_list[1].kwargs["messages"]
        == create.await_args_list[0].kwargs["messages"]
    )


@pytest.mark.parametrize(
    "failure", ["length", "network", "network_after_parse", "cancel", "filtered"]
)
async def test_other_stream_failures_are_not_argument_retried(hass, failure):
    if failure in {"network", "network_after_parse"}:
        stream = _Stream(
            *(
                [_chunk(_tool("{"), finish="tool_calls")]
                if failure == "network_after_parse"
                else []
            ),
            APIConnectionError(request=httpx.Request("POST", "https://example.test")),
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
@pytest.mark.parametrize(
    "error_type", [ParseArgumentsFailed, vol.Invalid, HomeAssistantError]
)
async def test_executor_errors_are_results_not_argument_retries(
    hass, after_correction, error_type
):
    streams = ([_calls(_tool("{"))] if after_correction else []) + [
        _calls(_tool("{}")),
        _answer(),
    ]
    entity, log, create = _setup(
        hass,
        streams,
        options={"max_function_calls_per_conversation": 2 if after_correction else 1},
    )
    entity._execute_function_tool.side_effect = error_type("private executor data")

    await _run(entity, log)

    assert create.await_count == len(streams)
    entity._execute_function_tool.assert_awaited_once()
    assert _results(log)[0].result.error
    assert _tool_result_data(_results(log)[0])["error"] == error_type.__name__
    assert _tool_result_data(_results(log)[0])["execution_status"] == "unknown"
    assert log.content[-1].content == "Finished."
    assert create.await_args.kwargs["tool_choice"] == "none"


@pytest.mark.parametrize(
    "invalid_repair", ["empty_response", "duplicate_id", "changed_id_arguments"]
)
async def test_invalid_correction_is_not_executed(hass, invalid_repair):
    repair = {
        "empty_response": _Stream(_chunk(finish="stop")),
        "duplicate_id": _calls(_tool("{}"), _tool("{}", index=1)),
        "changed_id_arguments": _calls(_tool('{"value":2}', call_id="completed")),
    }[invalid_repair]
    entity, log, create = _setup(
        hass,
        [_calls(_tool('{"value":1}', call_id="completed")), _calls(_tool("{")), repair],
        options={"max_function_calls_per_conversation": 3},
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
        options={"max_function_calls_per_conversation": 2, "temperature": 0.3},
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


@pytest.mark.parametrize(
    "calls",
    [
        [_tool("{", call_id=None)],
        [_tool("{", name=None)],
        [_tool("{}", call_id="")],
        [_tool("{}", name="")],
        [_tool("{}"), _tool("{", index=1)],
    ],
)
async def test_unpairable_call_retries_clean_history_without_invented_feedback(
    hass, calls
):
    entity, log, create = _setup(
        hass, [_calls(*calls), _calls(_tool("{}", call_id="real_retry_id")), _answer()]
    )

    await _run(entity, log)

    assert create.await_count == 3
    assert create.await_args_list[1].kwargs == create.await_args_list[0].kwargs
    assert [c.args[1].id for c in entity._execute_function_tool.await_args_list] == [
        "real_retry_id"
    ]
    assert [r.tool_call_id for r in _results(log)] == ["real_retry_id"]
    for request in create.await_args_list:
        _assert_paired_tools(request.kwargs["messages"])


async def test_model_may_answer_or_ask_for_clarification_after_tool_error(hass):
    entity, log, create = _setup(
        hass,
        [
            _calls(_tool("{")),
            _Stream(_chunk(text="Which device did you mean?", finish="stop")),
        ],
    )

    await _run(entity, log)

    assert create.await_count == 2
    entity._execute_function_tool.assert_not_awaited()
    assert log.content[-1].content == "Which device did you mean?"
    assert log.continue_conversation
    assert not log.unresponded_tool_results
    assert [c.role for c in log.content] == ["system", "user", "assistant"]
    assert [m["role"] for m in create.await_args_list[1].kwargs["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]


async def test_complete_failed_response_reports_all_errors_and_skipped_peers(hass):
    usage = SimpleNamespace(prompt_tokens=30, completion_tokens=5, total_tokens=35)
    failed = _Stream(
        _chunk(_tool("{", call_id="json_error"), finish="tool_calls"),
        _chunk(
            _tool("[]", call_id="type_error"),
            _tool("{}", index=1, call_id="not_run"),
            finish="tool_calls",
        ),
        _chunk(usage=usage),
    )
    entity, log, create = _setup(
        hass, [failed, _calls(_tool("{}", call_id="accepted")), _answer()]
    )

    await _run(entity, log)

    feedback = create.await_args_list[1].kwargs["messages"][2:]
    assert [m["role"] for m in feedback] == ["assistant", "tool", "tool", "tool"]
    errors = [json.loads(m["content"]) for m in feedback[1:]]
    assert errors[0]["error"] == "JSONDecodeError"
    with pytest.raises(vol.Invalid) as validator_error:
        vol.Schema(dict)([])
    assert errors[1] == {
        "error": type(validator_error.value).__name__,
        "error_text": str(validator_error.value),
    }
    assert errors[2] == {"status": "not_executed"}
    entity._execute_function_tool.assert_awaited_once()
    assert entity._execute_function_tool.await_args.args[1].id == "accepted"
    log.async_trace.assert_called_once_with(
        {"stats": {"input_tokens": 30, "output_tokens": 5}}
    )
    _assert_paired_tools(create.await_args_list[1].kwargs["messages"])


async def test_feedback_is_request_local_and_does_not_create_invalid_ha_entries(hass):
    broken = '{"secret":"private-argument"'
    entity, log, create = _setup(
        hass,
        [
            _calls(_tool(broken, call_id="bad")),
            _calls(_tool("{}", call_id="good")),
            _answer(),
        ],
    )

    await _run(entity, log)

    feedback_request = create.await_args_list[1].kwargs["messages"]
    assert feedback_request[-2]["tool_calls"][0]["function"]["arguments"] == broken
    assert "private-argument" not in str(log.as_dict())
    assert "private-argument" not in str(create.await_args_list[2].kwargs["messages"])
    assert [r.tool_call_id for r in _results(log)] == ["good"]
    assert all(
        isinstance(t.tool_args, dict)
        for c in log.content
        if c.role == "assistant"
        for t in c.tool_calls or []
    )
    for request in create.await_args_list:
        _assert_paired_tools(request.kwargs["messages"])


async def test_transport_failure_during_feedback_request_does_not_retry_again(hass):
    failed = _calls(_tool("{"))
    network_error = APIConnectionError(
        request=httpx.Request("POST", "https://example.test")
    )
    entity, log, create = _setup(hass, [failed, network_error])
    checkpoint = log.content.copy()

    with pytest.raises(APIConnectionError):
        await _run(entity, log)

    assert create.await_count == 2
    entity._execute_function_tool.assert_not_awaited()
    failed.close.assert_awaited_once()
    assert log.content == checkpoint
    _assert_paired_tools(create.await_args_list[1].kwargs["messages"])


async def test_real_sdk_sends_unmodified_argument_string_and_paired_error(hass):
    requests = []
    broken = '{"value":'

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        _assert_paired_tools(body["messages"])
        if len(requests) == 1:
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "bad",
                        "type": "function",
                        "function": {"name": "do_work", "arguments": broken},
                    }
                ]
            }
        elif len(requests) == 2:
            messages = body["messages"]
            assert [m["role"] for m in messages] == [
                "system",
                "user",
                "assistant",
                "tool",
            ]
            assert messages[-2]["tool_calls"][0]["function"]["arguments"] == broken
            assert json.loads(messages[-1]["content"]) == {
                "error": "JSONDecodeError",
                "error_text": "Expecting value: line 1 column 10 (char 9)",
            }
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "good",
                        "type": "function",
                        "function": {"name": "do_work", "arguments": '{"value":1}'},
                    }
                ]
            }
        else:
            delta = {"content": "Finished."}
        event = {
            "id": "response",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": "tool_calls" if "tool_calls" in delta else "stop",
                }
            ],
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
        )

    entity, log, _ = _setup(hass, [])
    async with AsyncOpenAI(
        api_key="test-key",
        base_url="https://example.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as client:
        entity.entry.runtime_data = client
        await _run(entity, log)

    assert len(requests) == 3
    entity._execute_function_tool.assert_awaited_once()
    assert log.content[-1].content == "Finished."


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
    assert [d["content"] for d in deltas if "content" in d] == ["Finished."]
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
    previous_messages = create.await_args_list[1].kwargs["messages"]
    assert (
        create.await_args_list[2].kwargs["messages"][: len(previous_messages)]
        == previous_messages
    )
    log.async_trace.assert_called_once_with(
        {"stats": {"input_tokens": 100, "output_tokens": 20}}
    )


@pytest.mark.parametrize("new_conversation", [False, True])
@pytest.mark.parametrize("failure_kind", ["json", "validator"])
async def test_terminal_error_and_diagnostics_are_redacted(
    hass, caplog, new_conversation, failure_kind
):
    secret = "private-token-device-command"
    broken = (
        json.dumps({"secret": secret})[:-1]
        if failure_kind == "json"
        else json.dumps(secret)
    )
    entity, log, create = _setup(
        hass,
        [
            _calls(_tool("{}", call_id="completed")),
            _calls(_tool(broken)),
            _calls(_tool(broken)),
        ],
        options={"max_function_calls_per_conversation": 3},
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
    assert "tool_rounds=3, tool_limit=3" in caplog.text
    error = ParseArgumentsFailed(broken)
    assert secret not in str(error)
    assert secret not in repr(error.args)
    assert secret not in repr(vars(error))


async def test_backend_rejecting_feedback_cannot_echo_sensitive_arguments(hass, caplog):
    secret = "private-backend-echo"
    response = httpx.Response(
        400, request=httpx.Request("POST", "https://example.test")
    )
    error = BadRequestError(
        f"invalid arguments: {secret}", response=response, body={"error": secret}
    )
    entity, log, create = _setup(
        hass,
        [_calls(_tool(json.dumps(secret))), error],
        agent_class=ExtendedOpenAIAgentEntity,
    )
    entity._get_exposed_entities = MagicMock(return_value=[])
    entity._get_function_tools = MagicMock(return_value=TOOLS)
    entity._build_system_prompt = MagicMock(return_value="System prompt")
    user_input = MagicMock(language="en", conversation_id=log.conversation_id)
    user_input.as_llm_context.return_value = None
    caplog.set_level(
        logging.DEBUG, logger="custom_components.extended_openai_conversation"
    )

    result = await entity._async_handle_message(user_input, log)

    assert create.await_count == 2
    entity._execute_function_tool.assert_not_awaited()
    assert secret not in str(result.as_dict())
    assert secret not in caplog.text
    assert "BadRequestError" in caplog.text
    assert [c.role for c in log.content] == ["system", "user", "assistant"]
    _assert_paired_tools(create.await_args_list[1].kwargs["messages"])
