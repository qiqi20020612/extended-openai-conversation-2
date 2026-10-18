"""Execution failures and replay protection using HA's real ChatLog/ToolResult."""

import asyncio
import errno
import json
import logging
import sqlite3
from unittest.mock import AsyncMock, MagicMock

from aiohttp import ClientConnectionError
import httpx
from openai import APIConnectionError
import probatio as vol
import pytest
from sqlalchemy.exc import SQLAlchemyError
import yaml

from custom_components.extended_openai_conversation.conversation import (
    ExtendedOpenAIAgentEntity,
)
from custom_components.extended_openai_conversation.entity import (
    MAX_TOOL_ITERATIONS,
    _convert_content_to_param,
)
from custom_components.extended_openai_conversation.exceptions import (
    EntityNotExposed,
    EntityNotFound,
    InvalidFunction,
)
from custom_components.extended_openai_conversation.functions import get_function
from homeassistant.components import conversation
from homeassistant.exceptions import (
    HomeAssistantError,
    ServiceNotFound,
    TemplateError,
    Unauthorized,
)
from homeassistant.helpers import llm
from tests.test_tool_argument_retry import (
    _answer,
    _assert_paired_tools,
    _calls,
    _chunk,
    _results,
    _setup,
    _Stream,
    _tool,
)


def tool_config(model_name="do_work", kind="template", **config):
    if kind == "template":
        config.setdefault("value_template", "ok")
    return {
        "spec": {
            "name": model_name,
            "parameters": {"type": "object", "properties": {}},
        },
        "function": get_function(kind).validate_schema({"type": kind, **config}),
    }


def actual_entity(hass, streams, *, options=None, agent_class=None):
    extra = {"agent_class": agent_class} if agent_class else {}
    entity, log, create = _setup(hass, streams, options=options, **extra)
    del entity._execute_function_tool
    return entity, log, create


def assert_paired(log, create):
    _assert_paired_tools(_convert_content_to_param(log.content))
    for request in create.await_args_list:
        _assert_paired_tools(request.kwargs["messages"])


@pytest.mark.parametrize(
    "error",
    [
        HomeAssistantError("private-token"),
        ServiceNotFound("private-token", "action"),
        EntityNotFound("private-token"),
        EntityNotExposed("private-token"),
        vol.Invalid("private-token"),
        KeyError("private-token"),
        TypeError("private-token"),
        ValueError("private-token"),
        sqlite3.OperationalError("private-token"),
        PermissionError("private-token"),
        TimeoutError("private-token"),
        ConnectionError("private-token"),
        ClientConnectionError("private-token"),
        httpx.ConnectError("private-token"),
        RuntimeError("private-token"),
        SQLAlchemyError("private-token"),
        yaml.YAMLError("private-token"),
        TemplateError(ValueError("private-token")),
        Unauthorized(),
    ],
)
async def test_expected_exceptions_are_safe_tool_results(
    hass, monkeypatch, caplog, error
):
    tools = [tool_config()]
    execute = AsyncMock(side_effect=error)
    monkeypatch.setattr(get_function("template"), "execute", execute)
    entity, log, create = actual_entity(hass, [_calls(_tool("{}")), _answer()])
    caplog.set_level(
        logging.DEBUG, logger="custom_components.extended_openai_conversation"
    )

    await entity._async_handle_chat_log(log, tools, [])

    execute.assert_awaited_once()
    assert create.await_count == 2
    result = _results(log)[0].result
    assert isinstance(result, llm.ToolResult) and result.error
    assert result.data["error"] == type(error).__name__
    assert "private-token" not in str(log.as_dict()) + caplog.text
    assert log.content[-1].content == "Finished."
    assert not log.unresponded_tool_results
    assert_paired(log, create)


async def test_partial_batch_keeps_successes_and_repairs_only_failed_tool(
    hass, monkeypatch
):
    execute = AsyncMock(
        side_effect=["first", ValueError("private-token"), "third", "repaired"]
    )
    monkeypatch.setattr(get_function("template"), "execute", execute)
    tools = [tool_config()]
    entity, log, create = actual_entity(
        hass,
        [
            _calls(
                *[
                    _tool(json.dumps({"value": n}), index=n, call_id=f"first_{n}")
                    for n in range(3)
                ]
            ),
            _calls(
                *[
                    _tool(json.dumps({"value": n}), index=n, call_id=f"new_{n}")
                    for n in range(3)
                ]
            ),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, tools, [])

    assert execute.await_count == 4
    results = _results(log)
    assert [r.result.error for r in results] == [
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert results[0].result.data == results[3].result.data
    assert results[2].result.data == results[5].result.data
    assert results[4].result.data == {"result": "repaired"}
    assert_paired(log, create)


@pytest.mark.parametrize("status", ["unknown", "partial"])
async def test_unconfirmed_operations_block_replay_and_parameter_rewrites(
    hass, monkeypatch, status
):
    outcome = llm.ToolResult(
        data={
            "execution_status": status,
            "retryable": False,
            "requires_state_check": True,
            **(
                {"error": "Unconfirmed"}
                if status != "scheduled"
                else {"result": "Scheduled"}
            ),
        },
        error=status != "scheduled",
    )
    execute = AsyncMock(side_effect=[outcome, "independent"])
    monkeypatch.setattr(get_function("template"), "execute", execute)
    tools = [tool_config(), tool_config("other_tool")]
    entity, log, create = actual_entity(
        hass,
        [
            _calls(
                _tool('{"value":1}', call_id="original"),
                _tool("{}", index=1, call_id="independent", name="other_tool"),
            ),
            _calls(
                _tool('{"value":1}', call_id="same_operation"),
                _tool('{"value":2}', index=1, call_id="rewritten"),
            ),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, tools, [])

    assert execute.await_count == 2
    results = _results(log)
    assert results[0].result == results[2].result
    assert results[1].result.data == {"result": "independent"}
    assert results[3].result.error
    assert results[3].result.data["error"] == "UnsafeRetryBlocked"
    assert_paired(log, create)


@pytest.mark.parametrize("reuse_id", [False, True])
async def test_repeated_success_is_only_reused_for_same_call_id(
    hass, monkeypatch, reuse_id
):
    execute = AsyncMock(return_value="ok")
    monkeypatch.setattr(get_function("template"), "execute", execute)
    tools = [tool_config()]
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool("{}", call_id="first")),
            _calls(_tool("{}", call_id="first" if reuse_id else "new_intent")),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, tools, [])

    assert execute.await_count == (1 if reuse_id else 2)
    assert len(_results(log)) == 2
    assert_paired(log, create)


@pytest.mark.parametrize("limit", [1, 2, 3, 10, -1])
async def test_execution_failures_share_existing_tool_budget(hass, monkeypatch, limit):
    rounds = MAX_TOOL_ITERATIONS if limit == -1 else limit
    execute = AsyncMock(side_effect=ValueError("private-token"))
    monkeypatch.setattr(get_function("template"), "execute", execute)
    entity, log, create = actual_entity(
        hass,
        [_calls(_tool("{}", call_id=f"call_{n}")) for n in range(rounds)] + [_answer()],
        options={"max_function_calls_per_conversation": limit},
    )

    await entity._async_handle_chat_log(log, [tool_config()], [])

    assert create.await_count == rounds + 1
    assert execute.await_count == rounds
    assert create.await_args.kwargs["tool_choice"] == "none"
    assert len(_results(log)) == rounds and all(r.result.error for r in _results(log))
    assert log.content[-1].content == "Finished."
    assert_paired(log, create)


@pytest.mark.parametrize(
    "error",
    [
        asyncio.CancelledError(),
        MemoryError(),
        RecursionError(),
        AssertionError(),
        ImportError("private-token"),
        OSError(errno.ENOSPC, "private-token"),
        InvalidFunction("private-token"),
    ],
)
async def test_fatal_or_cancelled_execution_pairs_remaining_calls_and_propagates(
    hass, monkeypatch, error
):
    execute = AsyncMock(side_effect=["completed", error])
    monkeypatch.setattr(get_function("template"), "execute", execute)
    entity, log, create = actual_entity(
        hass, [_calls(*[_tool("{}", index=n, call_id=f"call_{n}") for n in range(3)])]
    )

    with pytest.raises(type(error)):
        await entity._async_handle_chat_log(log, [tool_config()], [])

    assert execute.await_count == 2 and create.await_count == 1
    assert [r.result.data.get("execution_status") for r in _results(log)] == [
        None,
        "unknown",
        "not_executed",
    ]
    assert _results(log)[0].result.data == {"result": "completed"}
    assert "private-token" not in str(log.as_dict())
    assert_paired(log, create)


async def test_shutdown_does_not_start_remaining_tools(hass, monkeypatch):
    async def execute(*args):
        hass.is_stopping = True
        return "completed"

    call = AsyncMock(side_effect=execute)
    monkeypatch.setattr(get_function("template"), "execute", call)
    entity, log, create = actual_entity(
        hass, [_calls(_tool("{}", call_id="one"), _tool("{}", index=1, call_id="two"))]
    )

    with pytest.raises(HomeAssistantError):
        await entity._async_handle_chat_log(log, [tool_config()], [])

    call.assert_awaited_once()
    assert _results(log)[1].result.data["execution_status"] == "not_executed"
    assert_paired(log, create)


async def test_unknown_tool_and_missing_required_parameters_never_execute(
    hass, monkeypatch
):
    execute = AsyncMock(return_value="ok")
    monkeypatch.setattr(get_function("template"), "execute", execute)
    configured = tool_config()
    configured["spec"]["parameters"]["required"] = ["value"]
    entity, log, create = actual_entity(
        hass,
        [
            _calls(
                _tool("{}", name="missing_tool", call_id="unknown"),
                _tool("{}", index=1, call_id="no_value"),
            ),
            _calls(_tool('{"value":1}', call_id="corrected")),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, [configured], [])

    execute.assert_awaited_once()
    assert _results(log)[0].result.data["error"] == "FunctionNotFound"
    assert _results(log)[1].result.data["missing_parameters"] == ["value"]
    assert _results(log)[1].result.data["execution_status"] == "not_executed"
    assert_paired(log, create)


@pytest.mark.parametrize("native_error", [False, True])
async def test_native_tool_result_is_not_double_wrapped(
    hass, monkeypatch, native_error
):
    original = llm.ToolResult(
        data={"error": "SafeError", "execution_status": "not_executed"}
        if native_error
        else {"answer": 42},
        error=native_error,
    )
    monkeypatch.setattr(
        get_function("template"), "execute", AsyncMock(return_value=original)
    )
    entity, log, create = actual_entity(hass, [_calls(_tool("{}")), _answer()])

    await entity._async_handle_chat_log(log, [tool_config()], [])

    assert _results(log)[0].result == original
    assert_paired(log, create)


async def test_legacy_error_return_retains_payload_and_error_flag(hass, monkeypatch):
    original = {
        "error": "safe legacy error",
        "execution_status": "not_executed",
        "retryable": True,
    }
    monkeypatch.setattr(
        get_function("read_file"), "execute", AsyncMock(return_value=original)
    )
    entity, log, create = actual_entity(hass, [_calls(_tool("{}")), _answer()])

    await entity._async_handle_chat_log(
        log, [tool_config(kind="read_file", path="unused")], []
    )

    assert _results(log)[0].result.error
    assert _results(log)[0].result.data == original
    assert_paired(log, create)


@pytest.mark.parametrize("native", [False, True])
async def test_error_without_execution_metadata_cannot_replay_side_effects(
    hass, monkeypatch, native
):
    failure = {"error": "Already performed part of the operation"}
    execute = AsyncMock(
        return_value=llm.ToolResult(data=failure, error=True) if native else failure
    )
    monkeypatch.setattr(get_function("bash"), "execute", execute)
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool('{"value":1}', call_id="original")),
            _calls(_tool('{"value":2}', call_id="rewritten")),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(
        log, [tool_config(kind="bash", command="printf unused")], []
    )

    execute.assert_awaited_once()
    first = _results(log)[0].result
    assert first.error and first.data["error"] == failure["error"]
    assert first.data["execution_status"] == "unknown" and not first.data["retryable"]
    assert first.data["requires_state_check"]
    assert _results(log)[1].result.data["error"] == "UnsafeRetryBlocked"
    assert_paired(log, create)


@pytest.mark.parametrize("bad_result", ["str", "json"])
async def test_serialization_after_execution_never_reexecutes_operation(
    hass, monkeypatch, caplog, bad_result
):
    class BrokenString:
        def __str__(self):
            raise ValueError("private-token")

    result = (
        BrokenString()
        if bad_result == "str"
        else llm.ToolResult(data={"bad": object()})
    )
    execute = AsyncMock(return_value=result)
    monkeypatch.setattr(get_function("template"), "execute", execute)
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool("{}", call_id="one")),
            _calls(_tool("{}", call_id="two")),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, [tool_config()], [])

    execute.assert_awaited_once()
    assert _results(log)[0].result.data["error"] == "ResultSerializationError"
    assert _results(log)[0].result == _results(log)[1].result
    assert "private-token" not in caplog.text + str(log.as_dict())
    assert_paired(log, create)


@pytest.mark.parametrize("failure", ["json", "network", "close", "cancel"])
async def test_uncommitted_stream_has_no_listener_events_and_rolls_back(hass, failure):
    stream = _Stream(
        _chunk(text="Already completed private-token."),
        _chunk(_tool("{" if failure == "json" else "{}"), finish="tool_calls"),
    )
    if failure in {"network", "cancel"}:
        stream.chunks += (
            asyncio.CancelledError()
            if failure == "cancel"
            else APIConnectionError(
                request=httpx.Request("POST", "https://example.test")
            ),
        )
    if failure == "close":
        stream.close.side_effect = APIConnectionError(
            request=httpx.Request("POST", "https://example.test")
        )
    entity, log, create = actual_entity(
        hass, [stream], options={"max_function_calls_per_conversation": 1}
    )
    checkpoint = log.content.copy()

    with pytest.raises(
        (HomeAssistantError, APIConnectionError, asyncio.CancelledError)
    ):
        await entity._async_handle_chat_log(log, [tool_config()], [])

    assert log.content == checkpoint
    log.delta_listener.assert_not_called()
    stream.close.assert_awaited_once()
    assert create.await_count == 1


async def test_tool_result_listener_failure_does_not_break_batch(
    hass, monkeypatch, caplog
):
    execute = AsyncMock(return_value="ok")
    monkeypatch.setattr(get_function("template"), "execute", execute)
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool("{}", call_id="one"), _tool("{}", index=1, call_id="two")),
            _answer(),
        ],
    )

    def listener(log, delta):
        if delta.get("role") == "tool_result":
            raise ValueError("private-token")

    log.delta_listener.side_effect = listener

    await entity._async_handle_chat_log(log, [tool_config()], [])

    assert execute.await_count == 2
    assert len(_results(log)) == 2
    assert "private-token" not in caplog.text
    assert_paired(log, create)


async def test_static_configuration_and_api_failures_are_terminal_and_redacted(
    hass, monkeypatch, caplog
):
    entity, log, create = actual_entity(hass, [], agent_class=ExtendedOpenAIAgentEntity)
    entity._get_exposed_entities = MagicMock(return_value=[])
    entity._get_function_tools = MagicMock(side_effect=InvalidFunction("private-token"))
    user = MagicMock(language="en", conversation_id=None)
    user.as_llm_context.return_value = None

    first = await entity._async_handle_message(user, log)

    assert (
        first.response.error_code is not None
        and first.conversation_id == log.conversation_id
    )
    create.assert_not_awaited()
    entity._get_function_tools = MagicMock(return_value=[tool_config()])
    entity._build_system_prompt = MagicMock(return_value="System prompt")
    create.side_effect = [
        APIConnectionError(
            message="private-token",
            request=httpx.Request("POST", "https://example.test"),
        )
    ]
    second = await entity._async_handle_message(user, log)
    assert second.response.error_code is not None and create.await_count == 1
    assert "private-token" not in caplog.text + str(log.as_dict()) + str(
        first.as_dict()
    ) + str(second.as_dict())
    log.async_add_user_content(conversation.UserContent(content="Continue"))
    create.side_effect = [_answer()]
    third = await entity._async_handle_message(user, log)
    assert third.response.error_code is None
    assert log.content[-1].content == "Finished."


@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.ENOSPC, "private-token"),
        AttributeError("private-token"),
        sqlite3.DatabaseError("private-token"),
    ],
)
async def test_unrecoverable_errors_do_not_retry_the_model_or_disclose_values(
    hass, monkeypatch, caplog, error
):
    if isinstance(error, sqlite3.Error):
        error.sqlite_errorcode = sqlite3.SQLITE_IOERR_READ
    monkeypatch.setattr(
        get_function("template"), "execute", AsyncMock(side_effect=error)
    )
    entity, log, create = actual_entity(
        hass, [_calls(_tool("{}"))], agent_class=ExtendedOpenAIAgentEntity
    )
    entity._get_exposed_entities = MagicMock(return_value=[])
    entity._get_function_tools = MagicMock(return_value=[tool_config()])
    entity._build_system_prompt = MagicMock(return_value="System prompt")
    user = MagicMock(language="en", conversation_id=None)
    user.as_llm_context.return_value = None

    result = await entity._async_handle_message(user, log)

    assert result.response.error_code is not None
    assert create.await_count == 1
    assert _results(log)[0].result.data["execution_status"] == "unknown"
    assert (
        "private-token" not in str(log.as_dict()) + str(result.as_dict()) + caplog.text
    )
    assert_paired(log, create)


async def test_sqlite_extended_permission_error_cannot_be_bypassed_by_new_arguments(
    hass, monkeypatch
):
    error = sqlite3.OperationalError("private-token")
    error.sqlite_errorcode = sqlite3.SQLITE_READONLY_DBMOVED
    execute = AsyncMock(side_effect=error)
    monkeypatch.setattr(get_function("template"), "execute", execute)
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool("{}", call_id="first")),
            _calls(_tool('{"changed":true}', call_id="changed")),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, [tool_config()], [])

    execute.assert_awaited_once()
    assert _results(log)[0].result.data["permission_denied"]
    assert not _results(log)[0].result.data["retryable"]
    assert _results(log)[1].result.data["error"] == "UnsafeRetryBlocked"
    assert_paired(log, create)
