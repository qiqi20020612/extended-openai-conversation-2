"""Automation validation and writes must have different retry semantics."""

import asyncio
import builtins
from contextlib import suppress
import json
from threading import Barrier, BrokenBarrierError
from unittest.mock import AsyncMock

import probatio as vol
import pytest
import yaml

from custom_components.extended_openai_conversation.exceptions import InvalidFunction
from custom_components.extended_openai_conversation.functions import get_function
from homeassistant.exceptions import HomeAssistantError
from tests.test_tool_argument_retry import _answer, _calls, _results, _tool
from tests.test_tool_execution import actual_entity, assert_paired, tool_config

VALIDATOR = "custom_components.extended_openai_conversation.functions.native.automation.config._async_validate_config_item"


def automation_args(identifier="example"):
    return {
        "automation_config": yaml.safe_dump(
            {"id": identifier, "alias": "Example", "triggers": [], "actions": []}
        )
    }


@pytest.mark.parametrize("failure", ["missing", "yaml", "shape", "validation"])
async def test_invalid_automation_input_never_writes_or_reloads(
    hass, monkeypatch, tmp_path, caplog, failure
):
    validate = AsyncMock(side_effect=vol.Invalid("private-token"))
    monkeypatch.setattr(VALIDATOR, validate)
    arguments = {
        "missing": {},
        "yaml": {"automation_config": "private-token: ["},
        "shape": {"automation_config": "private-token"},
        "validation": automation_args(),
    }[failure]

    result = await get_function("native").add_automation(hass, {}, arguments, None, [])

    assert result.error and result.data["execution_status"] == "not_executed"
    assert result.data["retryable"]
    assert not (tmp_path / "automations.yaml").exists()
    hass.services.async_call.assert_not_awaited()
    assert "private-token" not in str(result) + caplog.text


async def test_validation_error_is_repaired_before_the_only_automation_write(
    hass, monkeypatch, tmp_path
):
    validate = AsyncMock(side_effect=[vol.Invalid("private-token"), {}])
    monkeypatch.setattr(VALIDATOR, validate)
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool(json.dumps(automation_args("invalid")), call_id="bad")),
            _calls(_tool(json.dumps(automation_args("fixed")), call_id="fixed")),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(
        log, [tool_config(kind="native", name="add_automation")], []
    )

    assert validate.await_count == 2
    assert _results(log)[0].result.data["execution_status"] == "not_executed"
    assert _results(log)[1].result.data == {"result": "Success"}
    assert [
        item["id"]
        for item in yaml.safe_load((tmp_path / "automations.yaml").read_text())
    ] == ["fixed"]
    hass.services.async_call.assert_awaited_once_with(
        "automation", "reload", blocking=True, context=None
    )
    assert_paired(log, create)


@pytest.mark.parametrize("failure", ["reload_error", "reload_timeout"])
async def test_reload_failure_after_write_is_unknown_and_cannot_create_again(
    hass, monkeypatch, tmp_path, caplog, failure
):
    validate = AsyncMock(return_value={})
    monkeypatch.setattr(VALIDATOR, validate)
    hass.services.async_call.side_effect = (
        HomeAssistantError("private-token")
        if failure == "reload_error"
        else TimeoutError("private-token")
    )
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool(json.dumps(automation_args("first")), call_id="first")),
            _calls(
                _tool(json.dumps(automation_args("duplicate")), call_id="rewritten")
            ),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(
        log, [tool_config(kind="native", name="add_automation")], []
    )

    validate.assert_awaited_once()
    hass.services.async_call.assert_awaited_once()
    assert [
        item["id"]
        for item in yaml.safe_load((tmp_path / "automations.yaml").read_text())
    ] == ["first"]
    first = _results(log)[0].result
    assert first.error and first.data["execution_status"] == "unknown"
    assert first.data["requires_state_check"] and not first.data["retryable"]
    assert _results(log)[1].result.data["error"] == "UnsafeRetryBlocked"
    hass.bus.async_fire.assert_not_called()
    assert "private-token" not in str(first) + caplog.text
    assert_paired(log, create)


async def test_duplicate_automation_id_never_appends_or_reloads(
    hass, monkeypatch, tmp_path
):
    monkeypatch.setattr(VALIDATOR, AsyncMock(return_value={}))
    path = tmp_path / "automations.yaml"
    original = yaml.safe_dump([{"id": "example", "alias": "Existing"}])
    path.write_text(original)

    result = await get_function("native").add_automation(
        hass, {}, automation_args(), None, []
    )

    assert result.error and result.data["execution_status"] == "not_executed"
    assert path.read_text() == original
    hass.services.async_call.assert_not_awaited()


@pytest.mark.parametrize("initial", [None, "", "[]\n"])
@pytest.mark.parametrize("same_id", [False, True])
async def test_concurrent_automation_storage_preserves_writes_and_checks_ids(
    hass, monkeypatch, tmp_path, initial, same_id
):
    monkeypatch.setattr(VALIDATOR, AsyncMock(return_value={}))
    path = tmp_path / "automations.yaml"
    if initial is not None:
        path.write_text(initial)
    original_open = builtins.open
    original_load = yaml.safe_load
    readers = Barrier(2)

    def overlap_empty_reads():
        # Without serialization both jobs see empty storage before either writes.
        # A serialized first reader times out here, then the next sees its write.
        with suppress(BrokenBarrierError):
            readers.wait(timeout=0.1)

    def checked_open(filename, *args, **kwargs):
        try:
            return original_open(filename, *args, **kwargs)
        except FileNotFoundError:
            if str(filename) == str(path):
                overlap_empty_reads()
            raise

    def checked_load(raw):
        loaded = original_load(raw)
        if raw in {"", "[]\n"}:
            overlap_empty_reads()
        return loaded

    monkeypatch.setattr(builtins, "open", checked_open)
    monkeypatch.setattr(yaml, "safe_load", checked_load)
    hass.async_add_executor_job.side_effect = asyncio.to_thread
    results = await asyncio.gather(
        *[
            get_function("native").add_automation(
                hass, {}, automation_args(identifier), None, []
            )
            for identifier in ["first", "first" if same_id else "second"]
        ]
    )

    stored = yaml.safe_load(path.read_text())
    assert sorted(item["id"] for item in stored) == (
        ["first"] if same_id else ["first", "second"]
    )
    if same_id:
        failed = [result for result in results if result != "Success"]
        assert len(failed) == 1
        assert failed[0].error
        assert failed[0].data["execution_status"] == "not_executed"
    else:
        assert results == ["Success", "Success"]
    assert hass.services.async_call.await_count == (1 if same_id else 2)


@pytest.mark.parametrize(
    "stored", ["invalid: [private-token", "private-token: invalid"]
)
async def test_corrupt_automation_storage_is_terminal(
    hass, monkeypatch, tmp_path, stored
):
    monkeypatch.setattr(VALIDATOR, AsyncMock(return_value={}))
    path = tmp_path / "automations.yaml"
    path.write_text(stored)

    with pytest.raises(InvalidFunction):
        await get_function("native").add_automation(
            hass, {}, automation_args(), None, []
        )

    assert path.read_text() == stored
    hass.services.async_call.assert_not_awaited()
