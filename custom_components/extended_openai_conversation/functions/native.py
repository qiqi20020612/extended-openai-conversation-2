"""Native tool for Home Assistant operations."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from datetime import timedelta
from functools import partial
import logging
import os
from threading import Lock
import time
from typing import Any

import probatio as vol
import yaml

from homeassistant.components import automation, energy, recorder
from homeassistant.components.recorder import history as recorder_history
from homeassistant.config import AUTOMATION_CONFIG_PATH
from homeassistant.const import SERVICE_RELOAD
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound
from homeassistant.helpers import llm, service as ha_service, target
import homeassistant.util.dt as dt_util

from ..const import EVENT_AUTOMATION_REGISTERED
from ..exceptions import (
    CallServiceError,
    EntityNotExposed,
    EntityNotFound,
    InvalidFunction,
    NativeNotFound,
)
from .base import Function
from .errors import (
    EXPECTED_TOOL_ERRORS,
    UNCONFIRMED_OPERATION_MESSAGE,
    execution_error,
    is_fatal_error,
)

_LOGGER = logging.getLogger(__name__)

# Bound the Assist wait. A timeout never proves that an action did not happen.
_SERVICE_CALL_TIMEOUT = 30

# Executor jobs can outlive the awaiting conversation, including cancellation.
# Hold a thread lock across the whole read/check/write transaction in the job.
_AUTOMATION_STORAGE_LOCK = Lock()


def service_call_arguments(
    arguments: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Normalize the supported service argument forms without mutating input."""
    domain = vol.Schema(str)(arguments["domain"])
    service = vol.Schema(str)(arguments["service"])
    data = deepcopy(
        vol.Schema(dict)(arguments.get("service_data", arguments.get("data", {})))
    )
    entity_ids = data.get("entity_id", arguments.get("entity_id"))
    if isinstance(entity_ids, str):
        entity_ids = [item.strip() for item in entity_ids.split(",") if item.strip()]
    if entity_ids is not None:
        data["entity_id"] = vol.Schema([str])(entity_ids)
    return domain, service, data


def service_target_entities(
    hass: HomeAssistant, domain: str, service: str, data: dict[str, Any]
) -> set[str]:
    """Use HA's effective target set for exposure and operation-overlap checks."""
    selection = target.TargetSelection(data)
    selected = target.async_extract_referenced_entity_ids(hass, selection)
    if any(
        (
            selected.missing_areas,
            selected.missing_devices,
            selected.missing_floors,
            selected.missing_labels,
        )
    ):
        raise EntityNotFound("target")
    all_entities = selected.referenced | selected.indirectly_referenced
    registered_service = hass.services.async_services()[domain][service]
    handler = registered_service.job.target
    # Platform service domains can differ from the domains of their entities.
    if isinstance(handler, partial) and handler.func in {
        ha_service.entity_service_call,
        ha_service.batched_entity_service_call,
    }:
        entities = (
            handler.args[1]
            if len(handler.args) > 1
            else handler.keywords.get("registered_entities")
        )
        if callable(entities):
            entities = entities()
        if isinstance(entities, Mapping):
            all_entities.intersection_update(entities)
    elif domain == "homeassistant" and service in {"turn_on", "turn_off", "toggle"}:
        all_entities = {
            entity_id
            for entity_id in all_entities
            if (entity_domain := entity_id.partition(".")[0]) != domain
            and hass.services.has_service(entity_domain, service)
        }
    # Unknown/custom handlers retain conservative checks of all targets.
    return all_entities


def service_operation_keys(
    hass: HomeAssistant, arguments: dict[str, Any]
) -> set[tuple[str, ...]]:
    """Identify the same action through aliases and overlapping target selectors."""
    domain, service, data = service_call_arguments(arguments)
    return {
        (
            "ha_service",
            entity_id.partition(".")[0]
            if domain == "homeassistant"
            and service in {"turn_on", "turn_off", "toggle"}
            else domain,
            service,
            entity_id,
        )
        for entity_id in service_target_entities(hass, domain, service, data)
    }


class NativeFunction(Function):
    def __init__(self) -> None:
        """Initialize native tool."""
        super().__init__(
            vol.Schema(
                {
                    vol.Required("name"): vol.In(
                        {
                            "execute_service",
                            "execute_service_single",
                            "add_automation",
                            "get_history",
                            "get_energy",
                            "get_statistics",
                            "get_user_from_user_id",
                        }
                    )
                }
            )
        )

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        name = function_config["name"]
        if name == "execute_service":
            return await self.execute_service(
                hass, function_config, arguments, llm_context, exposed_entities
            )
        if name == "execute_service_single":
            return await self.execute_service_single(
                hass, function_config, arguments, llm_context, exposed_entities
            )
        if name == "add_automation":
            return await self.add_automation(
                hass, function_config, arguments, llm_context, exposed_entities
            )
        if name == "get_history":
            return await self.get_history(
                hass, function_config, arguments, llm_context, exposed_entities
            )
        if name == "get_energy":
            return await self.get_energy(
                hass, function_config, arguments, llm_context, exposed_entities
            )
        if name == "get_statistics":
            return await self.get_statistics(
                hass, function_config, arguments, llm_context, exposed_entities
            )
        if name == "get_user_from_user_id":
            return await self.get_user_from_user_id(
                hass, function_config, arguments, llm_context, exposed_entities
            )

        raise NativeNotFound(name)

    async def execute_service_single(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        service_argument: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        started = False
        try:
            domain, service, service_data = service_call_arguments(service_argument)
            selection = target.TargetSelection(service_data)
            if not selection.has_any_target:
                raise CallServiceError(domain, service, {})
            if "all" in selection.entity_ids:
                raise EntityNotExposed("all")
            if not hass.services.has_service(domain, service):
                raise ServiceNotFound(domain, service)

            # Validate direct IDs and every expanded group/area/device target.
            # Preserve service-specific target fields and HA's dispatch semantics.
            self.validate_entity_ids(hass, list(selection.entity_ids), exposed_entities)
            all_entities = service_target_entities(hass, domain, service, service_data)
            registered_service = hass.services.async_services()[domain][service]
            if not all_entities:
                raise EntityNotFound("target")
            self.validate_entity_ids(hass, list(all_entities), exposed_entities)
            # HA validates again at dispatch. This preflight keeps schema failures
            # distinguishable from exceptions raised after the handler starts.
            schema = registered_service.schema
            if schema is not None:
                schema(deepcopy(service_data))
            started = True
            async with asyncio.timeout(_SERVICE_CALL_TIMEOUT):
                await hass.services.async_call(
                    domain=domain,
                    service=service,
                    service_data=service_data,
                    blocking=True,
                    context=llm_context.context if llm_context else None,
                )
            return {
                "success": True,
                "execution_status": "service_completed",
                "device_state_confirmed": False,
            }
        except EXPECTED_TOOL_ERRORS as err:
            if is_fatal_error(err) or hass.is_stopping:
                raise
            _LOGGER.warning(
                "HA service call failed (%s, dispatched=%s)",
                type(err).__name__,
                started,
            )
            return execution_error(
                err, status="unknown" if started else "not_executed"
            ).data

    async def execute_service(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | llm.ToolResult:
        result = []
        try:
            service_arguments = vol.Schema(vol.All([dict], vol.Length(min=1)))(
                arguments["list"]
            )
        except (KeyError, vol.Invalid) as err:
            return execution_error(err, status="not_executed")
        unconfirmed: set[tuple[str, ...]] = set()
        for service_argument in service_arguments:
            try:
                scope = service_operation_keys(hass, service_argument)
            except EXPECTED_TOOL_ERRORS as err:
                if is_fatal_error(err) or hass.is_stopping:
                    raise
                scope = set()  # The executor will return the actual input error.
            if scope & unconfirmed:
                item = {
                    "error": "UnsafeRetryBlocked",
                    "error_text": UNCONFIRMED_OPERATION_MESSAGE,
                    "execution_status": "not_executed",
                    "retryable": False,
                    "requires_state_check": True,
                }
            else:
                item = await self.execute_service_single(
                    hass,
                    function_config,
                    service_argument,
                    llm_context,
                    exposed_entities,
                )
            result.append(item)
            if item.get("execution_status") in {"unknown", "partial"}:
                unconfirmed.update(scope)
        if any("error" in item for item in result):
            partial = any(item.get("success") for item in result)
            status = (
                "partial"
                if partial
                else (
                    "unknown"
                    if any(item.get("execution_status") == "unknown" for item in result)
                    else "not_executed"
                )
            )
            failure_data: dict[str, Any] = {
                "error": "ServiceBatchFailed",
                "results": result,
                "execution_status": status,
                "retryable": status == "not_executed"
                and not any(item.get("permission_denied") for item in result),
                "permission_denied": any(
                    item.get("permission_denied") for item in result
                ),
                "requires_state_check": status in {"partial", "unknown"},
            }
            return llm.ToolResult(data=failure_data, error=True)
        return result

    async def add_automation(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> str | llm.ToolResult:
        write_started = False
        try:
            automation_config = yaml.safe_load(arguments["automation_config"])
            if isinstance(automation_config, list):
                automation_config = vol.Schema(vol.All([dict], vol.Length(min=1)))(
                    automation_config
                )[0]
            automation_config = vol.Schema(dict)(automation_config)
            config = {"id": str(round(time.time() * 1000)), **automation_config}
            await automation.config._async_validate_config_item(
                hass, config, True, False
            )

            def append_automation() -> str:
                nonlocal write_started
                path = os.path.join(hass.config.config_dir, AUTOMATION_CONFIG_PATH)
                try:
                    with open(path, encoding="utf-8") as current_file:
                        try:
                            current_automations = (
                                yaml.safe_load(current_file.read()) or []
                            )
                        except yaml.YAMLError as err:
                            raise InvalidFunction("automation storage") from err
                except FileNotFoundError:
                    current_automations = []
                if not isinstance(current_automations, list):
                    raise InvalidFunction("automation storage")
                if any(
                    isinstance(item, dict) and str(item.get("id")) == str(config["id"])
                    for item in current_automations
                ):
                    raise vol.Invalid("Automation ID already exists")
                raw_config = yaml.dump([config], allow_unicode=True, sort_keys=False)
                write_started = True
                with open(
                    path, "a" if current_automations else "w", encoding="utf-8"
                ) as output:
                    output.write("\n" + raw_config)
                return raw_config

            def store_automation() -> str:
                with _AUTOMATION_STORAGE_LOCK:
                    return append_automation()

            raw_config = await hass.async_add_executor_job(store_automation)
            async with asyncio.timeout(_SERVICE_CALL_TIMEOUT):
                await hass.services.async_call(
                    automation.config.DOMAIN,
                    SERVICE_RELOAD,
                    blocking=True,
                    context=llm_context.context if llm_context else None,
                )
            hass.bus.async_fire(
                EVENT_AUTOMATION_REGISTERED,
                {"automation_config": config, "raw_config": raw_config},
            )
            return "Success"
        except EXPECTED_TOOL_ERRORS as err:
            if is_fatal_error(err) or hass.is_stopping:
                raise
            _LOGGER.warning(
                "Automation creation failed (%s, write_started=%s)",
                type(err).__name__,
                write_started,
            )
            return execution_error(
                err, status="unknown" if write_started else "not_executed"
            )

    async def get_history(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        start_time = arguments.get("start_time")
        end_time = arguments.get("end_time")
        entity_ids = arguments.get("entity_ids", [])
        include_start_time_state = arguments.get("include_start_time_state", True)
        significant_changes_only = arguments.get("significant_changes_only", True)
        minimal_response = arguments.get("minimal_response", True)
        no_attributes = arguments.get("no_attributes", True)

        now = dt_util.utcnow()
        one_day = timedelta(days=1)
        start_time = self.as_utc(start_time, now - one_day, "start_time not valid")
        end_time = self.as_utc(end_time, start_time + one_day, "end_time not valid")

        self.validate_entity_ids(hass, entity_ids, exposed_entities)

        with recorder.util.session_scope(hass=hass, read_only=True) as session:
            result = await recorder.get_instance(hass).async_add_executor_job(
                recorder_history.get_significant_states_with_session,
                hass,
                session,
                start_time,
                end_time,
                entity_ids,
                None,
                include_start_time_state,
                significant_changes_only,
                minimal_response,
                no_attributes,
            )

        return [[self.as_dict(item) for item in sublist] for sublist in result.values()]

    async def get_energy(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        energy_manager: energy.data.EnergyManager = await energy.async_get_manager(hass)
        if energy_manager.data is None:
            return {}
        # energy_manager.data is EnergyPreferences which is a TypedDict (already a dict)
        return dict(energy_manager.data)

    async def get_user_from_user_id(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if (
            llm_context is None
            or llm_context.context is None
            or llm_context.context.user_id is None
        ):
            return {"name": "Unknown"}
        user = await hass.auth.async_get_user(llm_context.context.user_id)
        user_name = (
            user.name
            if user and hasattr(user, "name") and user.name is not None
            else "Unknown"
        )
        return {"name": user_name}

    async def get_statistics(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        statistic_ids = arguments.get("statistic_ids", [])
        start_time_parsed = dt_util.parse_datetime(arguments["start_time"])
        end_time_parsed = dt_util.parse_datetime(arguments["end_time"])
        if start_time_parsed is None or end_time_parsed is None:
            raise HomeAssistantError("Invalid datetime format")
        start_time = dt_util.as_utc(start_time_parsed)
        end_time = dt_util.as_utc(end_time_parsed)

        return await recorder.get_instance(hass).async_add_executor_job(
            recorder.statistics.statistics_during_period,
            hass,
            start_time,
            end_time,
            statistic_ids,
            arguments.get("period", "day"),
            arguments.get("units"),
            arguments.get("types", {"change"}),
        )

    def as_utc(
        self, value: str | None, default_value: Any, parse_error_message: str
    ) -> Any:
        if value is None:
            return default_value

        parsed_datetime = dt_util.parse_datetime(value)
        if parsed_datetime is None:
            raise HomeAssistantError(parse_error_message)

        return dt_util.as_utc(parsed_datetime)

    def as_dict(self, state: State | dict[str, Any]) -> dict[str, Any]:
        if isinstance(state, State):
            return state.as_dict()
        return state
