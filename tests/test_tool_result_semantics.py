"""Business data fields must not be mistaken for execution error envelopes."""

import json
import logging
from unittest.mock import AsyncMock

import pytest

from custom_components.extended_openai_conversation.functions import get_function
from tests.test_tool_argument_retry import _answer, _calls, _results, _tool
from tests.test_tool_execution import actual_entity, assert_paired, tool_config


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize(
    "column,value",
    [("error", "0"), ("error", "NULL"), ("success", "0"), ("exit_code", "2")],
)
async def test_sqlite_status_named_columns_remain_successful_data(
    hass, temp_db_path, single, column, value
):
    configured = tool_config(
        kind="sqlite",
        db_url=str(temp_db_path),
        query=f"SELECT {value} AS {column}",
        single=single,
    )
    entity, log, create = actual_entity(hass, [_calls(_tool("{}")), _answer()])

    await entity._async_handle_chat_log(log, [configured], [])

    row = {column: None if value == "NULL" else int(value)}
    assert not _results(log)[0].result.error
    assert _results(log)[0].result.data == {"result": str(row if single else [row])}
    assert_paired(log, create)


@pytest.mark.parametrize(
    "payload",
    [{"error": None}, {"error": "business data"}, {"success": False}, {"exit_code": 2}],
)
async def test_parsed_template_status_named_fields_remain_successful_data(
    hass, payload
):
    hass.config.legacy_templates = False
    configured = tool_config(value_template="{{ payload }}", parse_result=True)
    entity, log, create = actual_entity(
        hass, [_calls(_tool(json.dumps({"payload": payload}))), _answer()]
    )

    await entity._async_handle_chat_log(log, [configured], [])

    assert not _results(log)[0].result.error
    assert _results(log)[0].result.data == {"result": str(payload)}
    assert_paired(log, create)


@pytest.mark.parametrize("source", ["sqlite", "template"])
async def test_composite_continues_after_status_named_data(
    hass, tmp_path, temp_db_path, source
):
    hass.config.legacy_templates = False
    first = (
        {
            "type": "sqlite",
            "db_url": str(temp_db_path),
            "query": "SELECT 0 AS error",
            "single": True,
        }
        if source == "sqlite"
        else {
            "type": "template",
            "value_template": "{{ {'error': none} }}",
            "parse_result": True,
        }
    )
    configured = tool_config(
        kind="composite",
        sequence=[
            {**first, "response_variable": "row"},
            {"type": "write_file", "path": "after.txt", "content": "continued"},
            {"type": "template", "value_template": "{{ row }}", "parse_result": True},
        ],
    )
    entity, log, create = actual_entity(hass, [_calls(_tool("{}")), _answer()])

    await entity._async_handle_chat_log(log, [configured], [])

    assert (
        tmp_path / "extended_openai_conversation" / "after.txt"
    ).read_text() == "continued"
    assert not _results(log)[0].result.error
    assert _results(log)[0].result.data == {
        "result": str({"error": 0 if source == "sqlite" else None})
    }
    assert_paired(log, create)


async def test_background_success_with_error_named_data_is_not_logged_as_failure(
    hass, caplog
):
    hass.config.legacy_templates = False
    configured = tool_config(
        kind="composite",
        sequence=[
            {
                "type": "template",
                "value_template": "{{ {'error': none} }}",
                "parse_result": True,
            }
        ],
    )
    entity, _, _ = actual_entity(hass, [])
    caplog.set_level(
        logging.DEBUG, logger="custom_components.extended_openai_conversation"
    )

    await entity._async_run_background_tool(configured["function"], {}, None, [])

    assert "status=completed" in caplog.text
    assert "failed_or_unknown" not in caplog.text


@pytest.mark.parametrize(
    "kind,config,payload",
    [
        ("script", {"sequence": []}, {"error": None}),
        ("native", {"name": "get_energy"}, {"success": False}),
        ("rest", {"resource": "https://example.test"}, {"exit_code": 2}),
    ],
)
async def test_other_data_producers_do_not_inherit_legacy_error_conventions(
    hass, monkeypatch, kind, config, payload
):
    monkeypatch.setattr(get_function(kind), "execute", AsyncMock(return_value=payload))
    entity, log, create = actual_entity(hass, [_calls(_tool("{}")), _answer()])

    await entity._async_handle_chat_log(log, [tool_config(kind=kind, **config)], [])

    assert not _results(log)[0].result.error
    assert _results(log)[0].result.data == {"result": str(payload)}
    assert_paired(log, create)
