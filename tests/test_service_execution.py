"""Service execution contract against HA 2026.10's real registry, without devices."""

import asyncio
from copy import deepcopy
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import probatio as vol
import pytest

from custom_components.extended_openai_conversation.functions import (
    NativeFunction,
    get_function,
)
from homeassistant.components import homeassistant as homeassistant_component
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    config_validation as cv,
    device_registry as dr,
    entity_registry as er,
    llm,
    service as ha_service,
)
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.entity_platform import DATA_DOMAIN_PLATFORM_ENTITIES
from tests.test_tool_argument_retry import _answer, _calls, _results, _tool
from tests.test_tool_execution import actual_entity, assert_paired, tool_config


@pytest.fixture
async def service_hass(tmp_path):
    instance = HomeAssistant(str(tmp_path))
    instance.states.async_set("light.allowed", "off")
    instance.states.async_set("light.private", "off")
    yield instance
    await instance.async_stop(force=True)


def service_args(entity="light.allowed", **data):
    return {
        "domain": "light",
        "service": "turn_on",
        "service_data": {"entity_id": entity, **data},
    }


EXPOSED = [{"entity_id": "light.allowed"}]


@pytest.mark.parametrize("selector", ["area_id", "device_id"])
@pytest.mark.parametrize("service_domain", ["light", "custom", "homeassistant"])
async def test_entity_service_ignores_unrelated_domain_but_rejects_affected_private_entity(
    service_hass, selector, service_domain
):
    await ar.async_load(service_hass, load_empty=True)
    dr.async_setup(service_hass)
    await dr.async_load(service_hass, load_empty=True)
    await er.async_load(service_hass, load_empty=True)
    service_hass.config_entries = MagicMock()
    service_hass.config_entries.async_get_entry.return_value = MagicMock(
        domain="test", entry_id="test", title="Test", pref_disable_new_entities=False
    )
    area = ar.async_get(service_hass).async_create("Mixed area")
    devices = dr.async_get(service_hass)
    device = devices.async_get_or_create(
        config_entry_id="test", identifiers={("test", "mixed")}
    )
    devices.async_update_device(device.id, area_id=area.id)
    registry = er.async_get(service_hass)
    for entity_id in ["light.allowed", "light.private", "sensor.private"]:
        service_hass.states.async_remove(entity_id)
        domain, name = entity_id.split(".")
        entry = registry.async_get_or_create(
            domain,
            "light" if domain == "light" else "test",
            f"{domain}_{name}",
            suggested_object_id=name,
        )
        registry.async_update_entity(
            entry.entity_id,
            area_id=area.id if selector == "area_id" else None,
            device_id=device.id if selector == "device_id" else None,
        )
        if domain == "sensor":
            service_hass.states.async_set(entity_id, "42")
    calls = []

    class TestLight(Entity):
        _attr_should_poll = False

        async def async_turn_on(self, **kwargs):
            calls.append(self.entity_id)

    component = EntityComponent(logging.getLogger(__name__), "light", service_hass)
    component.register_shutdown()
    lights = []
    for entity_id in ["light.allowed", "light.private"]:
        light = TestLight()
        light.entity_id = entity_id
        light._attr_unique_id = entity_id.replace(".", "_")
        lights.append(light)
    await component.async_add_entities(lights)
    if selector == "device_id":
        for light in lights:
            registry.async_update_entity(light.entity_id, device_id=device.id)
    component.async_register_entity_service("turn_on", {}, "async_turn_on")
    if service_domain == "custom":
        # Platform services may have a service domain different from entity domain.
        service_hass.data[DATA_DOMAIN_PLATFORM_ENTITIES][("light", "custom")] = {
            light.entity_id: light for light in lights
        }
        ha_service.async_register_platform_entity_service(
            service_hass,
            service_domain="custom",
            service_name="turn_on",
            entity_domain="light",
            func="async_turn_on",
            schema={},
        )
    elif service_domain == "homeassistant":
        assert await homeassistant_component.async_setup(service_hass, {})
    arguments = {
        "domain": service_domain,
        "service": "turn_on",
        "service_data": {selector: area.id if selector == "area_id" else device.id},
    }

    # Both lights can be affected, so the private light must still deny dispatch.
    denied = await NativeFunction().execute_service_single(
        service_hass, {}, arguments, None, EXPOSED
    )
    assert denied["permission_denied"] and not calls
    registry.async_update_entity("light.private", area_id=None, device_id=None)

    # The private sensor remains in the target but cannot be affected by light.turn_on.
    completed = await NativeFunction().execute_service_single(
        service_hass, {}, arguments, None, EXPOSED
    )
    assert completed["execution_status"] == "service_completed"
    assert calls == ["light.allowed"]
    assert arguments["service_data"] == {
        selector: area.id if selector == "area_id" else device.id
    }

    # A custom handler can affect multiple domains; do not infer its scope from
    # its service-domain name or drop the sensor without HA entity registration.
    custom = AsyncMock()
    service_hass.services.async_register("light", "wide_action", custom)
    denied = await NativeFunction().execute_service_single(
        service_hass,
        {},
        {**arguments, "domain": "light", "service": "wide_action"},
        None,
        EXPOSED,
    )
    assert denied["permission_denied"]
    custom.assert_not_awaited()

    if service_domain == "homeassistant":
        # Once a domain provides the action, the generic service could affect it.
        sensor = AsyncMock()
        service_hass.services.async_register("sensor", "turn_on", sensor)
        denied = await NativeFunction().execute_service_single(
            service_hass, {}, arguments, None, EXPOSED
        )
        assert denied["permission_denied"]
        sensor.assert_not_awaited()


async def test_real_ha_automation_validation_failure_never_writes(
    service_hass, tmp_path
):
    # Exercise HA's validator itself, without registering any automation or device.
    result = await NativeFunction().add_automation(
        service_hass,
        {},
        {"automation_config": "alias: Invalid\ntriggers: []\nactions: invalid"},
        None,
        [],
    )

    assert result.error and result.data["execution_status"] == "not_executed"
    assert result.data["retryable"] and not (tmp_path / "automations.yaml").exists()


async def test_actual_ha_script_partial_failure_is_not_replayed(
    hass, service_hass, monkeypatch, caplog
):
    monkeypatch.setattr(cv._hass, "hass", service_hass)
    effects = []

    async def record(call):
        effects.append("completed")

    async def fail(call):
        raise HomeAssistantError("private-token")

    service_hass.services.async_register("test", "record", record)
    service_hass.services.async_register("test", "fail", fail)
    configured = tool_config(
        kind="script",
        sequence=[
            {"action": "test.record"},
            {"action": "test.fail"},
            {"action": "test.record"},
        ],
    )
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool("{}", call_id="first")),
            _calls(_tool('{"changed":true}', call_id="rewritten")),
            _answer(),
        ],
    )
    entity.hass = service_hass

    await entity._async_handle_chat_log(log, [configured], [])

    assert effects == ["completed"]
    assert _results(log)[0].result.data["execution_status"] == "unknown"
    assert _results(log)[1].result.data["error"] == "UnsafeRetryBlocked"
    assert "private-token" not in str(log.as_dict()) + caplog.text
    assert_paired(log, create)


async def test_service_result_waits_for_handler_and_does_not_claim_device_state(
    service_hass,
):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def handler(call):
        calls.append(call)
        entered.set()
        await release.wait()

    service_hass.services.async_register("light", "turn_on", handler)
    context = llm.LLMContext(
        platform="test",
        context=Context(),
        language="en",
        assistant="conversation",
        device_id=None,
    )
    arguments = service_args()
    original = deepcopy(arguments)
    task = asyncio.create_task(
        NativeFunction().execute_service_single(
            service_hass, {}, arguments, context, EXPOSED
        )
    )
    await entered.wait()

    assert not task.done()
    release.set()
    result = await task

    assert result == {
        "success": True,
        "execution_status": "service_completed",
        "device_state_confirmed": False,
    }
    assert calls[0].context is context.context
    assert arguments == original


@pytest.mark.parametrize(
    "failure",
    [
        "missing_parameter",
        "invalid_type",
        "missing_service",
        "missing_entity",
        "not_exposed",
        "schema",
    ],
)
async def test_service_preflight_errors_never_dispatch(service_hass, failure):
    handler = AsyncMock()
    service_hass.services.async_register(
        "light",
        "turn_on",
        handler,
        schema=vol.Schema(
            {vol.Required("entity_id"): [str], vol.Optional("brightness"): int}
        ),
    )
    arguments = service_args()
    if failure == "missing_parameter":
        arguments.pop("domain")
    elif failure == "invalid_type":
        arguments["service_data"] = []
    elif failure == "missing_service":
        arguments["service"] = "does_not_exist"
    elif failure == "missing_entity":
        arguments = service_args("light.absent")
    elif failure == "not_exposed":
        arguments = service_args("light.private")
    else:
        arguments = service_args(brightness="invalid")

    result = await NativeFunction().execute_service_single(
        service_hass, {}, arguments, None, EXPOSED
    )

    handler.assert_not_awaited()
    assert result["execution_status"] == "not_executed"
    assert "error" in result
    if failure == "not_exposed":
        assert result["permission_denied"] and not result["retryable"]


async def test_service_handler_failure_is_unknown_and_redacted(service_hass, caplog):
    effects = []

    async def handler(call):
        effects.append("already changed")
        raise HomeAssistantError("private-token-password")

    service_hass.services.async_register("light", "turn_on", handler)

    result = await NativeFunction().execute_service_single(
        service_hass, {}, service_args(), None, EXPOSED
    )

    assert effects == ["already changed"]
    assert result["execution_status"] == "unknown" and not result["retryable"]
    assert result["requires_state_check"] and "success" not in result
    assert "private-token-password" not in str(result) + caplog.text


async def test_service_timeout_is_unknown_not_unexecuted(service_hass, monkeypatch):
    effects = []

    async def handler(call):
        effects.append("already changed")
        await asyncio.Event().wait()

    service_hass.services.async_register("light", "turn_on", handler)
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation.functions.native._SERVICE_CALL_TIMEOUT",
        0.01,
    )

    result = await NativeFunction().execute_service_single(
        service_hass, {}, service_args(), None, EXPOSED
    )

    assert effects == ["already changed"]
    assert result["error"] == "TimeoutError"
    assert result["execution_status"] == "unknown" and result["requires_state_check"]
    assert not result["retryable"]


async def test_service_cancellation_propagates(service_hass):
    entered = asyncio.Event()

    async def handler(call):
        entered.set()
        await asyncio.Event().wait()

    service_hass.services.async_register("light", "turn_on", handler)
    task = asyncio.create_task(
        NativeFunction().execute_service_single(
            service_hass, {}, service_args(), None, EXPOSED
        )
    )
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_group_target_cannot_bypass_exposure(service_hass):
    handler = AsyncMock()
    service_hass.services.async_register("light", "turn_on", handler)
    service_hass.states.async_set(
        "group.mixed", "off", {"entity_id": ["light.allowed", "light.private"]}
    )
    exposed = [*EXPOSED, {"entity_id": "group.mixed"}]

    result = await NativeFunction().execute_service_single(
        service_hass, {}, service_args("group.mixed"), None, exposed
    )

    assert result["permission_denied"]
    handler.assert_not_awaited()


async def test_area_target_is_validated_without_changing_service_semantics(
    service_hass,
):
    await ar.async_load(service_hass, load_empty=True)
    dr.async_setup(service_hass)
    await dr.async_load(service_hass, load_empty=True)
    await er.async_load(service_hass, load_empty=True)
    area = ar.async_get(service_hass).async_create("Test area")
    registry = er.async_get(service_hass)
    entries = []
    for name in ["allowed", "private"]:
        service_hass.states.async_remove(f"light.{name}")
        entry = registry.async_get_or_create(
            "light", "test", name, suggested_object_id=name
        )
        registry.async_update_entity(entry.entity_id, area_id=area.id)
        service_hass.states.async_set(entry.entity_id, "off")
        entries.append(entry)
    handler = AsyncMock()
    service_hass.services.async_register("light", "turn_on", handler)
    arguments = {
        "domain": "light",
        "service": "turn_on",
        "service_data": {"area_id": area.id},
    }

    denied = await NativeFunction().execute_service_single(
        service_hass, {}, arguments, None, EXPOSED
    )

    assert denied["permission_denied"]
    handler.assert_not_awaited()
    registry.async_update_entity(entries[1].entity_id, area_id=None)
    completed = await NativeFunction().execute_service_single(
        service_hass, {}, arguments, None, EXPOSED
    )
    assert completed["execution_status"] == "service_completed"
    assert handler.await_args.args[0].data == {"area_id": area.id}
    assert arguments["service_data"] == {"area_id": area.id}


async def test_native_batch_retains_mixed_results_and_does_not_repeat(service_hass):
    calls = []

    async def handler(call):
        calls.append(dict(call.data))
        if call.data.get("fail"):
            raise HomeAssistantError("private-token")

    service_hass.services.async_register("light", "turn_on", handler)
    arguments = {"list": [service_args(), service_args(fail=True), service_args()]}
    result = await NativeFunction().execute_service(
        service_hass, {}, arguments, None, EXPOSED
    )

    assert isinstance(result, llm.ToolResult) and result.error
    assert result.data["execution_status"] == "partial"
    assert [item.get("success", False) for item in result.data["results"]] == [
        True,
        False,
        True,
    ]
    assert len(calls) == 3 and not result.data["retryable"]


async def test_real_service_failure_then_model_rewrite_is_not_reexecuted(
    hass, service_hass
):
    effects = []

    async def handler(call):
        effects.append(dict(call.data))
        raise HomeAssistantError("private-token")

    service_hass.services.async_register("light", "turn_on", handler)
    tools = [
        {
            "spec": {
                "name": "do_work",
                "parameters": {"type": "object", "properties": {}},
            },
            "function": get_function("native").validate_schema(
                {"type": "native", "name": "execute_service_single"}
            ),
        }
    ]
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool(json.dumps(service_args()), call_id="first")),
            _calls(_tool(json.dumps(service_args(brightness=10)), call_id="rewrite")),
            _answer(),
        ],
    )
    entity.hass = service_hass

    await entity._async_handle_chat_log(log, tools, EXPOSED)

    assert len(effects) == 1
    assert _results(log)[0].result.data["execution_status"] == "unknown"
    assert _results(log)[1].result.data["error"] == "UnsafeRetryBlocked"
    assert_paired(log, create)
