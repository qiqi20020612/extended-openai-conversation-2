"""Bash tool for shell command execution."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from pathlib import Path
import re
import signal

import probatio as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, llm

from ..const import (
    DEFAULT_WORKING_DIRECTORY,
    SHELL_DENY_PATTERNS,
    SHELL_OUTPUT_LIMIT,
    SHELL_TIMEOUT,
)
from .base import Function
from .errors import EXPECTED_TOOL_ERRORS, execution_error, is_fatal_error

_LOGGER = logging.getLogger(__name__)


class _CommandBlocked(PermissionError):
    """A policy rejection with a fixed message safe to preserve for compatibility."""


class BashFunction(Function):
    """Execute shell commands with security controls."""

    def get_working_dir(self, hass: HomeAssistant) -> Path:
        """Get the default working directory for bash operations."""
        return Path(hass.config.config_dir) / DEFAULT_WORKING_DIRECTORY

    def __init__(self) -> None:
        """Initialize bash tool."""
        schema = vol.Schema(
            {
                vol.Required("command"): cv.template,
                vol.Optional("cwd"): cv.template,
                vol.Optional("restrict_to_workspace", default=True): bool,
                vol.Optional("allow_patterns"): vol.All(cv.ensure_list, [str]),
            }
        )
        super().__init__(schema)

    def _guard_command(
        self,
        command: str,
        cwd: str | Path,
        restrict_to_workspace: bool,
        allow_patterns: list[str] | None = None,
    ) -> None:
        """Validate command against security policies."""
        # Deny patterns check
        for pattern in SHELL_DENY_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                raise _CommandBlocked("Command blocked by security policy")

        # Allow patterns check
        if allow_patterns:
            lower = command.lower()
            if not any(re.search(p, lower) for p in allow_patterns):
                raise _CommandBlocked("Command blocked: not in allowlist")

        # Path restriction check when restrict_to_workspace is enabled
        if restrict_to_workspace:
            # Block path traversal patterns
            if "../" in command or "..\\" in command:
                raise _CommandBlocked("Command blocked: path traversal detected")

            # Extract and validate paths in command
            win_paths = re.findall(r"[A-Za-z]:\\[^\\\"\' ]+", command)
            posix_paths = re.findall(r"(?<!\w)/[^\s\"\']+", command)

            for raw in win_paths + posix_paths:
                try:
                    p = Path(raw).resolve()
                except (OSError, RuntimeError) as err:
                    raise _CommandBlocked(
                        "Command blocked: path cannot be resolved"
                    ) from err

                if cwd not in p.parents and p != cwd:
                    raise _CommandBlocked(
                        "Command blocked: path outside working directory"
                    )

    async def execute(
        self,
        hass: HomeAssistant,
        function_config,
        arguments,
        llm_context: llm.LLMContext | None,
        exposed_entities,
    ):
        """Execute shell command with security controls."""
        # Security validation
        try:
            command_template = function_config.get("command")
            command = command_template.async_render(arguments, parse_result=False)
            cwd_template = function_config.get("cwd")
            cwd = (
                Path(cwd_template.async_render(arguments, parse_result=False))
                if cwd_template
                else self.get_working_dir(hass)
            )
            restrict_to_workspace = function_config.get("restrict_to_workspace", True)
            allow_patterns = function_config.get("allow_patterns", [])
            timeout = float(arguments.get("timeout", SHELL_TIMEOUT))
            if not math.isfinite(timeout) or not 0 < timeout <= SHELL_TIMEOUT:
                raise ValueError("Invalid command timeout")
            cwd = cwd.resolve()
            if restrict_to_workspace and not cwd.is_relative_to(
                self.get_working_dir(hass).resolve()
            ):
                raise _CommandBlocked(
                    "Command blocked: working directory outside workspace"
                )
            self._guard_command(
                command,
                cwd=cwd,
                restrict_to_workspace=restrict_to_workspace,
                allow_patterns=allow_patterns,
            )
        except EXPECTED_TOOL_ERRORS as err:
            if is_fatal_error(err) or hass.is_stopping:
                raise
            data = execution_error(err, status="not_executed").data
            if isinstance(err, _CommandBlocked):
                data["error"] = str(err)
            return data

        process = None
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except TimeoutError as err:
                return execution_error(err).data

            # Decode output with truncation
            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

            # Truncate output if too large
            if len(stdout_text) > SHELL_OUTPUT_LIMIT:
                stdout_text = (
                    stdout_text[:SHELL_OUTPUT_LIMIT]
                    + "\n... (truncated, output too large)"
                )
            if len(stderr_text) > SHELL_OUTPUT_LIMIT:
                stderr_text = (
                    stderr_text[:SHELL_OUTPUT_LIMIT]
                    + "\n... (truncated, output too large)"
                )

            result = {
                "exit_code": process.returncode,
                "stdout": stdout_text,
            }

            if stderr_text:
                result["stderr"] = stderr_text
            if process.returncode:
                result.update(
                    execution_status="unknown",
                    retryable=False,
                    requires_state_check=True,
                )

        except EXPECTED_TOOL_ERRORS as err:
            if is_fatal_error(err):
                raise
            _LOGGER.warning("Shell operation failed (%s)", type(err).__name__)
            return execution_error(
                err, status="unknown" if process else "not_executed"
            ).data
        finally:
            # Killing only the shell can leave its child performing the operation.
            # Reap the process on timeout/cancellation; completed effects remain unknown.
            if process is not None and process.returncode is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
                await process.communicate()

        return result
