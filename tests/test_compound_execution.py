"""Partial composite effects and honest delayed-task outcomes."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock

import pytest

from custom_components.extended_openai_conversation.functions import get_function
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from tests.test_tool_argument_retry import _answer, _calls, _results, _tool
from tests.test_tool_execution import actual_entity, assert_paired, tool_config


@pytest.mark.parametrize("earlier_success", [False, True])
@pytest.mark.parametrize("failure_kind", ["exception", "legacy", "native"])
async def test_composite_stops_on_error_and_reports_each_step(
    hass, monkeypatch, earlier_success, failure_kind
):
    config = get_function("composite").validate_schema(
        {
            "type": "composite",
            "sequence": [{"type": "template", "value_template": "ok"}]
            * (3 if earlier_success else 2),
        }
    )
    failure = {
        "exception": ValueError("private-token"),
        "legacy": {
            "error": "safe legacy error",
            "execution_status": "not_executed",
            "retryable": True,
        },
        "native": llm.ToolResult(
            data={
                "error": "safe native error",
                "execution_status": "not_executed",
                "retryable": True,
            },
            error=True,
        ),
    }[failure_kind]
    execute = AsyncMock(
        side_effect=(["first completed"] if earlier_success else []) + [failure]
    )
    monkeypatch.setattr(get_function("template"), "execute", execute)

    result = await get_function("composite").execute(hass, config, {}, None, [])

    assert isinstance(result, llm.ToolResult) and result.error
    assert execute.await_count == int(earlier_success) + 1
    assert result.data["steps"][-1]["execution_status"] == "not_executed"
    assert "private-token" not in str(result)
    if earlier_success:
        assert result.data["execution_status"] == "partial"
        assert result.data["steps"][0]["result"] == "first completed"
        assert result.data["requires_state_check"] and not result.data["retryable"]
    else:
        assert result.data["retryable"]


async def test_partial_composite_cannot_repeat_file_write_after_correcting_arguments(
    hass, monkeypatch, tmp_path
):
    configured = tool_config(
        kind="composite",
        sequence=[
            {"type": "write_file", "path": "effect.txt", "content": "first effect"},
            {"type": "template", "value_template": "{{ 1 / value }}"},
            {"type": "write_file", "path": "never.txt", "content": "must not run"},
        ],
    )
    write = get_function("write_file")
    execute = AsyncMock(wraps=write.execute)
    monkeypatch.setattr(write, "execute", execute)
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool('{"value":0}', call_id="partial")),
            _calls(_tool('{"value":1}', call_id="corrected")),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, [configured], [])

    execute.assert_awaited_once()
    assert (
        tmp_path / "extended_openai_conversation" / "effect.txt"
    ).read_text() == "first effect"
    assert not (tmp_path / "extended_openai_conversation" / "never.txt").exists()
    first = _results(log)[0].result
    assert first.error and first.data["execution_status"] == "partial"
    assert [step["execution_status"] for step in first.data["steps"]] == [
        "completed",
        "failed",
        "not_executed",
    ]
    assert _results(log)[1].result.data["error"] == "UnsafeRetryBlocked"
    assert_paired(log, create)


async def test_delayed_failure_is_observed_without_rewriting_scheduled_result(
    hass, monkeypatch, caplog
):
    monkeypatch.setattr(
        get_function("script"), "execute", AsyncMock(return_value="delay complete")
    )
    execute = AsyncMock(side_effect=HomeAssistantError("private-token-password"))
    monkeypatch.setattr(get_function("template"), "execute", execute)
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool(json.dumps({"delay": {"seconds": 1}}), call_id="scheduled")),
            _calls(_tool(json.dumps({"delay": {"seconds": 1}}), call_id="duplicate")),
            _answer(),
        ],
    )
    tasks = []
    entity.entry.async_create_task.side_effect = lambda hass, coroutine: (
        tasks.append(asyncio.create_task(coroutine)) or tasks[-1]
    )
    caplog.set_level(
        logging.DEBUG, logger="custom_components.extended_openai_conversation"
    )

    await entity._async_handle_chat_log(log, [tool_config()], [])
    before = log.as_dict()
    await asyncio.gather(*tasks)

    assert len(tasks) == 1
    execute.assert_awaited_once()
    assert log.as_dict() == before
    assert _results(log)[0].result.data["execution_status"] == "scheduled"
    assert not _results(log)[0].result.data["completion_confirmed"]
    assert _results(log)[0].result == _results(log)[1].result
    assert (
        "failed_or_unknown" in caplog.text
        and "private-token-password" not in caplog.text
    )
    assert_paired(log, create)


async def test_delayed_cancellation_is_not_swallowed(hass, monkeypatch):
    entered = asyncio.Event()

    async def wait(*args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(get_function("script"), "execute", AsyncMock(side_effect=wait))
    entity, log, _ = actual_entity(
        hass, [_calls(_tool('{"delay":{"seconds":1}}')), _answer()]
    )
    tasks = []
    entity.entry.async_create_task.side_effect = lambda hass, coroutine: (
        tasks.append(asyncio.create_task(coroutine)) or tasks[-1]
    )

    await entity._async_handle_chat_log(log, [tool_config()], [])
    await entered.wait()
    tasks[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await tasks[0]
    assert _results(log)[0].result.data["execution_status"] == "scheduled"


async def test_script_refusing_to_run_is_not_success(hass, monkeypatch):
    script = AsyncMock()
    script.async_run.return_value = None
    constructor = monkeypatch.setattr
    constructor(
        "custom_components.extended_openai_conversation.functions.script.Script",
        lambda *args, **kwargs: script,
    )

    result = await get_function("script").execute(hass, {"sequence": []}, {}, None, [])

    assert result.error and result.data["execution_status"] == "not_executed"
    assert not result.data["retryable"]


async def test_scheduling_failure_never_claims_scheduled(hass):
    entity, log, create = actual_entity(
        hass, [_calls(_tool('{"delay":{"seconds":1}}')), _answer()]
    )
    entity.entry.async_create_task.side_effect = RuntimeError("private-token")

    await entity._async_handle_chat_log(log, [tool_config()], [])

    result = _results(log)[0].result
    assert result.error and result.data["execution_status"] == "unknown"
    assert "Scheduled" not in str(result)
    assert_paired(log, create)


async def test_invalid_delay_is_rejected_before_scheduling_and_can_be_corrected(hass):
    entity, log, create = actual_entity(
        hass,
        [
            _calls(
                _tool('{"delay":{"seconds":"invalid-private-token"}}', call_id="bad")
            ),
            _calls(_tool("{}", call_id="corrected")),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, [tool_config()], [])

    entity.entry.async_create_task.assert_not_called()
    result = _results(log)[0].result
    assert result.error and result.data["execution_status"] == "not_executed"
    assert result.data["retryable"] and "private-token" not in str(result)
    assert _results(log)[1].result.data == {"result": "ok"}
    assert_paired(log, create)
