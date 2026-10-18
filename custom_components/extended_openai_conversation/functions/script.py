"""Script tool for Home Assistant script sequences."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.script import config as script_config
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.script import Script

from ..const import DOMAIN
from .base import Function
from .errors import execution_error

_LOGGER = logging.getLogger(__name__)


class ScriptFunction(Function):
    def __init__(self) -> None:
        """Initialize script tool."""
        super().__init__(script_config.SCRIPT_ENTITY_SCHEMA)

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        script = Script(
            hass,
            function_config["sequence"],
            "extended_openai_conversation",
            DOMAIN,
            running_description="[extended_openai_conversation] function",
            logger=_LOGGER,
            log_exceptions=False,
        )

        context = llm_context.context if llm_context else None
        result = await script.async_run(run_variables=arguments, context=context)
        if result is None:
            # HA returns None when a run was refused (shutdown/mode/recursion).
            return execution_error(
                RuntimeError(), status="not_executed", retryable=False
            )
        return result.variables.get("_function_result", "Success")
