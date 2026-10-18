"""Action/target replay guards with real HA service dispatch and ChatLog."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from custom_components.extended_openai_conversation.const import (
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
)
from homeassistant.components import conversation
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)
from tests.test_service_execution import service_args, service_hass as _service_hass
from tests.test_tool_argument_retry import _answer, _calls, _results, _tool
from tests.test_tool_execution import actual_entity, assert_paired, tool_config

EXPOSED = [{"entity_id": f"light.{name}"} for name in ["allowed", "other", "third"]]

service_hass = _service_hass


def service_tool(name="do_work", *, batch=False):
    return tool_config(
        name,
        kind="native",
        name="execute_service" if batch else "execute_service_single",
    )


def call(arguments, identifier, *, index=0, name="do_work"):
    return _tool(json.dumps(arguments), call_id=identifier, index=index, name=name)


async def test_unknown_member_blocks_overlap_inside_native_batch_but_not_other_device(
    hass, service_hass
):
    service_hass.states.async_set("light.other", "off")
    effects = []

    async def handler(request):
        effects.append(dict(request.data))
        if request.data["entity_id"] == ["light.allowed"]:
            raise TimeoutError("private-token")

    service_hass.services.async_register("light", "turn_on", handler)
    entity, log, create = actual_entity(
        hass,
        [
            _calls(
                call(
                    {
                        "list": [
                            service_args(),
                            service_args(brightness=10),
                            service_args("light.other"),
                        ]
                    },
                    "batch",
                )
            ),
            _answer(),
        ],
    )
    entity.hass = service_hass

    await entity._async_handle_chat_log(log, [service_tool(batch=True)], EXPOSED)

    assert [effect["entity_id"] for effect in effects] == [
        ["light.allowed"],
        ["light.other"],
    ]
    result = _results(log)[0].result
    assert result.error and result.data["execution_status"] == "partial"
    assert result.data["results"][1]["error"] == "UnsafeRetryBlocked"
    assert result.data["results"][2]["success"]
    assert_paired(log, create)


async def test_different_action_through_same_tool_is_not_a_retry(hass, service_hass):
    effects = []
    queries = []

    async def control(request):
        effects.append(dict(request.data))
        raise TimeoutError("private-token")

    async def query(request):
        queries.append(service_hass.states.get("light.allowed").state)

    service_hass.services.async_register("light", "turn_on", control)
    service_hass.services.async_register("light", "query_state", query)
    query_args = {**service_args(), "service": "query_state"}
    entity, log, create = actual_entity(
        hass,
        [
            _calls(call(service_args(), "first")),
            _calls(call(query_args, "query")),
            _calls(call(service_args(brightness=10), "retry")),
            _answer(),
        ],
    )
    entity.hass = service_hass

    await entity._async_handle_chat_log(log, [service_tool()], EXPOSED)

    assert len(effects) == 1 and queries == ["off"]
    assert not _results(log)[1].result.error
    assert _results(log)[2].result.data["error"] == "UnsafeRetryBlocked"
    assert_paired(log, create)


@pytest.mark.parametrize("same_response", [False, True])
@pytest.mark.parametrize("failure", ["handler_error", "timeout"])
async def test_unknown_service_allows_other_device_but_blocks_same_action(
    hass, service_hass, monkeypatch, caplog, same_response, failure
):
    service_hass.states.async_set("light.other", "off")
    effects = []

    async def handler(request):
        effects.append(dict(request.data))
        if request.data["entity_id"] == ["light.allowed"]:
            if failure == "timeout":
                await asyncio.Event().wait()
            raise HomeAssistantError("private-token")

    monkeypatch.setattr(
        "custom_components.extended_openai_conversation.functions.native._SERVICE_CALL_TIMEOUT",
        0.01,
    )
    service_hass.services.async_register("light", "turn_on", handler)
    first = call(service_args(), "first")
    later = [
        call(service_args(brightness=10), "overlap", index=1 if same_response else 0),
        call(
            service_args("light.other"), "independent", index=2 if same_response else 1
        ),
    ]
    streams = (
        [_calls(first, *later)] if same_response else [_calls(first), _calls(*later)]
    )
    entity, log, create = actual_entity(hass, [*streams, _answer()])
    entity.hass = service_hass

    await entity._async_handle_chat_log(log, [service_tool()], EXPOSED)

    assert [effect["entity_id"] for effect in effects] == [
        ["light.allowed"],
        ["light.other"],
    ]
    results = [content.result for content in _results(log)]
    assert results[0].data["execution_status"] == "unknown"
    assert results[0].data["retryable"] is False
    assert results[0].data["requires_state_check"]
    assert "may already have executed" in results[0].data["recovery_action"]
    assert "Query the actual state first" in results[0].data["recovery_action"]
    assert results[1].data["error"] == "UnsafeRetryBlocked"
    assert results[1].data["execution_status"] == "not_executed"
    assert "Do not retry it directly" in results[1].data["error_text"]
    assert not results[2].error
    assert "private-token" not in str(log.as_dict()) + caplog.text
    assert log.content[-1].content == "Finished."
    assert_paired(log, create)


async def test_state_query_is_allowed_without_unlocking_unknown_operation(
    hass, service_hass
):
    effects = []

    async def handler(request):
        effects.append(dict(request.data))
        service_hass.states.async_set("light.allowed", "on")
        raise TimeoutError("private-token")

    service_hass.services.async_register("light", "turn_on", handler)
    query = tool_config("query_state", value_template="{{ states('light.allowed') }}")
    query["function"]["value_template"].hass = service_hass
    entity, log, create = actual_entity(
        hass,
        [
            _calls(call(service_args(), "first")),
            _calls(call({}, "query", name="query_state")),
            _calls(call(service_args(brightness=100), "unsafe_retry")),
            _answer(),
        ],
    )
    entity.hass = service_hass

    await entity._async_handle_chat_log(log, [service_tool(), query], EXPOSED)

    assert len(effects) == 1
    assert _results(log)[1].result.data == {"result": "on"}
    assert _results(log)[2].result.data["error"] == "UnsafeRetryBlocked"
    assert_paired(log, create)


@pytest.mark.parametrize("target_form", ["list", "comma", "outer", "data"])
async def test_overlapping_target_spellings_cannot_replay_completed_part(
    hass, service_hass, target_form
):
    for name in ["other", "third"]:
        service_hass.states.async_set(f"light.{name}", "off")
    effects = []

    async def handler(request):
        effects.append(dict(request.data))
        if "light.allowed" in request.data["entity_id"]:
            raise HomeAssistantError("private-token")

    service_hass.services.async_register("light", "turn_on", handler)
    overlap = service_args(["light.other", "light.third"])
    if target_form == "comma":
        overlap["service_data"]["entity_id"] = "light.other, light.third"
    elif target_form == "outer":
        overlap["entity_id"] = overlap["service_data"].pop("entity_id")
    elif target_form == "data":
        overlap["data"] = overlap.pop("service_data")
    entity, log, create = actual_entity(
        hass,
        [
            _calls(call(service_args(["light.allowed", "light.other"]), "first")),
            _calls(
                call(overlap, "overlap"),
                call(service_args("light.third"), "independent", index=1),
            ),
            _answer(),
        ],
    )
    entity.hass = service_hass

    await entity._async_handle_chat_log(log, [service_tool()], EXPOSED)

    assert len(effects) == 2
    assert effects[1]["entity_id"] == ["light.third"]
    assert _results(log)[1].result.data["error"] == "UnsafeRetryBlocked"
    assert_paired(log, create)


@pytest.mark.parametrize("selector", ["area_id", "device_id", "floor_id", "label_id"])
async def test_expanded_targets_overlap_direct_entities(hass, service_hass, selector):
    await ar.async_load(service_hass, load_empty=True)
    dr.async_setup(service_hass)
    await dr.async_load(service_hass, load_empty=True)
    await er.async_load(service_hass, load_empty=True)
    service_hass.config_entries = MagicMock()
    service_hass.config_entries.async_get_entry.return_value = MagicMock(
        domain="test", entry_id="test", title="Test", pref_disable_new_entities=False
    )
    area = ar.async_get(service_hass).async_create("Test area")
    if selector == "floor_id":
        from homeassistant.helpers import floor_registry as fr

        await fr.async_load(service_hass, load_empty=True)
        floor = fr.async_get(service_hass).async_create("Test floor", level=0)
        ar.async_get(service_hass).async_update(area.id, floor_id=floor.floor_id)
        target_id = floor.floor_id
    elif selector == "label_id":
        from homeassistant.helpers import label_registry as lr

        await lr.async_load(service_hass, load_empty=True)
        target_id = lr.async_get(service_hass).async_create("Test label").label_id
    else:
        target_id = area.id
    devices = dr.async_get(service_hass)
    device = devices.async_get_or_create(
        config_entry_id="test", identifiers={("test", "test")}
    )
    devices.async_update_device(device.id, area_id=area.id)
    if selector == "device_id":
        target_id = device.id
    registry = er.async_get(service_hass)
    service_hass.states.async_remove("light.allowed")
    entry = registry.async_get_or_create(
        "light", "test", "allowed", suggested_object_id="allowed"
    )
    registry.async_update_entity(
        entry.entity_id,
        device_id=device.id,
        labels={target_id} if selector == "label_id" else set(),
    )
    service_hass.states.async_set(entry.entity_id, "off")
    service_hass.states.async_set("light.other", "off")
    effects = []

    async def handler(request):
        effects.append(dict(request.data))
        if selector in request.data:
            raise TimeoutError("private-token")

    service_hass.services.async_register("light", "turn_on", handler)
    entity, log, create = actual_entity(
        hass,
        [
            _calls(
                call(
                    {
                        "domain": "light",
                        "service": "turn_on",
                        "service_data": {selector: target_id},
                    },
                    "expanded",
                )
            ),
            _calls(
                call(service_args(), "overlap"),
                call(service_args("light.other"), "independent", index=1),
            ),
            _answer(),
        ],
    )
    entity.hass = service_hass

    await entity._async_handle_chat_log(log, [service_tool()], EXPOSED)

    assert len(effects) == 2
    assert _results(log)[0].result.data["execution_status"] == "unknown"
    assert _results(log)[1].result.data["error"] == "UnsafeRetryBlocked"
    assert not _results(log)[2].result.error
    assert_paired(log, create)


async def test_partial_native_batch_protects_success_and_unknown_but_allows_new_targets(
    hass, service_hass
):
    for name in ["other", "third"]:
        service_hass.states.async_set(f"light.{name}", "off")
    effects = []

    async def handler(request):
        effects.append(dict(request.data))
        if request.data["entity_id"] == ["light.other"]:
            raise HomeAssistantError("private-token")

    service_hass.services.async_register("light", "turn_on", handler)
    entity, log, create = actual_entity(
        hass,
        [
            _calls(
                call({"list": [service_args(), service_args("light.other")]}, "partial")
            ),
            _calls(
                call({"list": [service_args(brightness=1)]}, "completed_overlap"),
                call(
                    {"list": [service_args("light.other", brightness=1)]},
                    "unknown_overlap",
                    index=1,
                ),
                call({"list": [service_args("light.third")]}, "independent", index=2),
            ),
            _answer(),
        ],
    )
    entity.hass = service_hass

    await entity._async_handle_chat_log(log, [service_tool(batch=True)], EXPOSED)

    assert len(effects) == 3
    assert _results(log)[0].result.data["execution_status"] == "partial"
    assert _results(log)[0].result.data["results"][0]["success"]
    assert [_results(log)[i].result.data["error"] for i in [1, 2]] == [
        "UnsafeRetryBlocked"
    ] * 2
    assert not _results(log)[3].result.error
    assert_paired(log, create)


async def test_partial_batch_member_rejected_before_dispatch_can_be_fixed(
    hass, service_hass
):
    service_hass.states.async_set("light.other", "off")
    effects = []

    async def handler(request):
        effects.append(dict(request.data))

    def schema(data):
        if "invalid" in data:
            raise ValueError("private-token")
        return data

    service_hass.services.async_register("light", "turn_on", handler, schema=schema)
    entity, log, create = actual_entity(
        hass,
        [
            _calls(
                call(
                    {
                        "list": [
                            service_args(),
                            service_args("light.other", invalid=True),
                        ]
                    },
                    "partial",
                )
            ),
            _calls(call({"list": [service_args("light.other")]}, "fixed")),
            _answer(),
        ],
    )
    entity.hass = service_hass

    await entity._async_handle_chat_log(log, [service_tool(batch=True)], EXPOSED)

    assert [effect["entity_id"] for effect in effects] == [
        ["light.allowed"],
        ["light.other"],
    ]
    assert _results(log)[0].result.data["execution_status"] == "partial"
    assert not _results(log)[1].result.error
    assert_paired(log, create)


@pytest.mark.parametrize("original_batch", [False, True])
@pytest.mark.parametrize("alias_batch", [False, True])
async def test_service_tool_alias_cannot_bypass_action_target_guard(
    hass, service_hass, original_batch, alias_batch
):
    effects = []

    async def handler(request):
        effects.append(dict(request.data))
        raise HomeAssistantError("private-token")

    service_hass.services.async_register("light", "turn_on", handler)
    entity, log, create = actual_entity(
        hass,
        [
            _calls(
                call(
                    {"list": [service_args()]} if original_batch else service_args(),
                    "first",
                )
            ),
            _calls(
                call(
                    {"list": [service_args(brightness=5)]}
                    if alias_batch
                    else service_args(brightness=5),
                    "alias",
                    name="alias",
                )
            ),
            _answer(),
        ],
    )
    entity.hass = service_hass

    await entity._async_handle_chat_log(
        log,
        [service_tool(batch=original_batch), service_tool("alias", batch=alias_batch)],
        EXPOSED,
    )

    assert len(effects) == 1
    assert _results(log)[1].result.data["error"] == "UnsafeRetryBlocked"
    assert_paired(log, create)


@pytest.mark.parametrize("budget", [1, 2, 3, 10, -1])
@pytest.mark.parametrize("still_exposed", [False, True])
async def test_next_user_turn_confirmation_resets_guard_and_shared_budget(
    hass, service_hass, budget, still_exposed
):
    effects = []

    async def handler(request):
        effects.append(dict(request.data))
        if len(effects) == 1:
            raise TimeoutError("private-token")

    service_hass.services.async_register("light", "turn_on", handler)
    entity, log, create = actual_entity(
        hass,
        [_calls(call(service_args(), "first")), _answer()],
        options={CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: budget},
    )
    entity.hass = service_hass
    tools = [service_tool()]
    await entity._async_handle_chat_log(log, tools, EXPOSED)
    assert _results(log)[0].result.data["execution_status"] == "unknown"
    assert create.await_args_list[1].kwargs["tool_choice"] == (
        "none" if budget == 1 else "auto"
    )

    log.async_add_user_content(
        conversation.UserContent(content="I checked the state. Please try again.")
    )
    create.side_effect = [_calls(call(service_args(), "confirmed")), _answer()]
    await entity._async_handle_chat_log(log, tools, EXPOSED if still_exposed else [])

    assert len(effects) == (2 if still_exposed else 1)
    assert _results(log)[1].result.error is (not still_exposed)
    if not still_exposed:
        assert _results(log)[1].result.data["permission_denied"]
    assert log.content[-1].content == "Finished."
    assert_paired(log, create)
