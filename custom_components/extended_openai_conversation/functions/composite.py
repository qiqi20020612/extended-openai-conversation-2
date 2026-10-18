"""Composite tool for chaining multiple functions."""

from __future__ import annotations

from typing import Any

import probatio as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, llm

from .base import Function
from .errors import (
    EXPECTED_TOOL_ERRORS,
    execution_error,
    function_is_read_only,
    is_fatal_error,
    result_is_error,
)


class CompositeFunction(Function):
    def __init__(self) -> None:
        """Initialize composite tool."""
        super().__init__(
            vol.Schema(
                {
                    vol.Required("sequence"): vol.All(
                        cv.ensure_list, [self.function_schema], vol.Length(min=1)
                    )
                }
            )
        )

    def function_schema(self, function_config: Any) -> dict[str, Any]:
        """Validate a composite function schema."""
        from . import get_function

        if not isinstance(function_config, dict):
            raise vol.Invalid("expected dictionary")

        composite_schema = {vol.Optional("response_variable"): str}
        function = get_function(function_config["type"])

        return dict(function.data_schema.extend(composite_schema)(function_config))

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        from . import get_function

        sequence = function_config["sequence"]
        new_arguments = arguments.copy()

        steps: list[dict[str, Any]] = []
        result: Any = None
        for index, next_function_config in enumerate(sequence):
            next_function = get_function(next_function_config["type"])
            try:
                result = await next_function.execute(
                    hass,
                    next_function_config,
                    new_arguments,
                    llm_context,
                    exposed_entities,
                )
            except EXPECTED_TOOL_ERRORS as err:
                if is_fatal_error(err) or hass.is_stopping:
                    raise
                result = execution_error(
                    err,
                    status="failed"
                    if function_is_read_only(next_function_config)
                    else "unknown",
                )
            failed = result_is_error(result, next_function_config)
            data = result.data if isinstance(result, llm.ToolResult) else result
            steps.append(
                {
                    "index": index,
                    "execution_status": (
                        data.get("execution_status", "unknown")
                        if failed and isinstance(data, dict)
                        else "unknown"
                        if failed
                        else "completed"
                    ),
                    "result": data,
                }
            )
            if failed:
                steps.extend(
                    {"index": skipped, "execution_status": "not_executed"}
                    for skipped in range(index + 1, len(sequence))
                )
                status = "partial" if index else steps[0]["execution_status"]
                failure_data: dict[str, Any] = {
                    "error": "CompositeStepFailed",
                    "steps": steps,
                    "execution_status": status,
                    "retryable": not index
                    and isinstance(data, dict)
                    and bool(data.get("retryable")),
                    "permission_denied": isinstance(data, dict)
                    and bool(data.get("permission_denied")),
                    "requires_state_check": status in {"partial", "unknown"},
                }
                return llm.ToolResult(data=failure_data, error=True)

            response_variable = next_function_config.get("response_variable")
            if response_variable:
                new_arguments[response_variable] = result

        return result
