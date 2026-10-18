"""File functions for read, write, and edit operations."""

from __future__ import annotations

from functools import partial
import logging
from pathlib import Path
from typing import Any

import probatio as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, llm
from homeassistant.helpers.template import Template

from ..const import (
    DEFAULT_ALLOWED_DIRS,
    DEFAULT_WORKING_DIRECTORY,
    FILE_READ_SIZE_LIMIT,
)
from .base import Function
from .errors import EXPECTED_TOOL_ERRORS, execution_error, is_fatal_error

_LOGGER = logging.getLogger(__name__)


def _file_error(error: Exception, status: str) -> dict[str, Any]:
    """Keep the legacy permission-error label without echoing an OS path."""
    data = execution_error(error, status=status).data
    if isinstance(error, PermissionError):
        data["error"] = "Access denied"
    return data


class FileFunction(Function):
    """Base class for file-related functions."""

    def get_working_dir(self, hass: HomeAssistant) -> Path:
        """Get the default working directory for file operations."""
        return Path(hass.config.config_dir) / DEFAULT_WORKING_DIRECTORY

    def to_absolute_path(
        self, hass: HomeAssistant, path: str, base_dir: Path | None = None
    ) -> Path:
        """Convert path to absolute path."""
        p = Path(path)
        if p.is_absolute():
            return p

        if base_dir is None:
            base_dir = Path(hass.config.config_dir)

        return base_dir / p

    def _resolve_path(
        self,
        hass: HomeAssistant,
        path: str,
        allow_dirs: list[str],
    ) -> Path:
        """Resolve path relative to working directory."""
        workdir = self.get_working_dir(hass)
        target = self.to_absolute_path(hass, path, workdir).resolve()

        # Check against allowed directories (already resolved to absolute paths)
        allowed = False
        for allow_dir in allow_dirs:
            allowed_path = Path(allow_dir).resolve()

            if target.is_relative_to(allowed_path):
                allowed = True
                break

        if not allowed:
            raise PermissionError("Access denied: path is not in allowed directories")

        return target

    def _render_allow_dirs(
        self,
        hass: HomeAssistant,
        allow_dirs: list[Template],
        arguments: dict[str, Any],
    ) -> list[str]:
        """Render allow_dir templates."""
        # Always include default allowed directories (resolved to absolute paths)
        all_allow_dirs = [
            str(self.to_absolute_path(hass, d)) for d in DEFAULT_ALLOWED_DIRS
        ]

        # Add custom allow_dir if specified
        if allow_dirs:
            template_arguments = {
                **arguments,
                "config_dir": hass.config.config_dir,
            }
            custom_dirs = [
                template.async_render(template_arguments, parse_result=False)
                for template in allow_dirs
            ]
            all_allow_dirs.extend(custom_dirs)

        return all_allow_dirs


class ReadFileFunction(FileFunction):
    """Read file contents."""

    def __init__(self) -> None:
        """Initialize read file tool."""
        schema = vol.Schema(
            {
                vol.Required("path"): cv.template,
                vol.Optional("allow_dir"): vol.All(cv.ensure_list, [cv.template]),
            }
        )
        super().__init__(schema)

    async def execute(
        self,
        hass: HomeAssistant,
        function_config,
        arguments,
        llm_context: llm.LLMContext | None,
        exposed_entities,
    ):
        """Read file contents."""
        try:
            path_template = function_config.get("path")
            path_str = path_template.async_render(arguments, parse_result=False)
            allow_dirs = self._render_allow_dirs(
                hass, function_config.get("allow_dir", []), arguments
            )
            target_path = self._resolve_path(hass, path_str, allow_dirs)

            if not target_path.exists():
                return {
                    "error": "File not found",
                    "execution_status": "not_executed",
                    "retryable": True,
                }

            if not target_path.is_file():
                return {
                    "error": "Not a file",
                    "execution_status": "not_executed",
                    "retryable": True,
                }

            # Check file size
            file_size = target_path.stat().st_size
            if file_size > FILE_READ_SIZE_LIMIT:
                return {
                    "error": f"File too large: {file_size} bytes (limit: {FILE_READ_SIZE_LIMIT})",
                    "execution_status": "not_executed",
                    "retryable": True,
                }

            # Read file
            content = await hass.async_add_executor_job(
                partial(target_path.read_text, encoding="utf-8")
            )

        except EXPECTED_TOOL_ERRORS as err:
            if is_fatal_error(err):
                raise
            _LOGGER.warning("File read failed (%s)", type(err).__name__)
            return _file_error(err, "failed")

        return {"content": content, "size": file_size}


class WriteFileFunction(FileFunction):
    """Write content to file."""

    def __init__(self) -> None:
        """Initialize write file tool."""
        schema = vol.Schema(
            {
                vol.Required("path"): cv.template,
                vol.Required("content"): cv.template,
                vol.Optional("allow_dir"): vol.All(cv.ensure_list, [cv.template]),
            }
        )
        super().__init__(schema)

    async def execute(
        self,
        hass: HomeAssistant,
        function_config,
        arguments,
        llm_context: llm.LLMContext | None,
        exposed_entities,
    ):
        """Write content to file."""
        write_started = False
        try:
            path_template = function_config.get("path")
            path_str = path_template.async_render(arguments, parse_result=False)
            content_template = function_config.get("content")
            content = content_template.async_render(arguments, parse_result=False)
            allow_dirs = self._render_allow_dirs(
                hass, function_config.get("allow_dir", []), arguments
            )
            target_path = self._resolve_path(hass, path_str, allow_dirs)

            # Write file
            write_started = True
            await hass.async_add_executor_job(
                partial(target_path.write_text, content, encoding="utf-8")
            )

            bytes_written = len(content.encode("utf-8"))

        except EXPECTED_TOOL_ERRORS as err:
            if is_fatal_error(err):
                raise
            _LOGGER.warning("File write failed (%s)", type(err).__name__)
            return _file_error(err, "unknown" if write_started else "not_executed")

        return {
            "success": True,
            "path": str(target_path),
            "bytes_written": bytes_written,
        }


class EditFileFunction(FileFunction):
    """Edit file with find-and-replace."""

    def __init__(self) -> None:
        """Initialize edit file tool."""
        schema = vol.Schema(
            {
                vol.Required("path"): cv.template,
                vol.Required("old_text"): cv.template,
                vol.Required("new_text"): cv.template,
                vol.Optional("allow_dir"): vol.All(cv.ensure_list, [cv.template]),
            }
        )
        super().__init__(schema)

    async def execute(
        self,
        hass: HomeAssistant,
        function_config,
        arguments,
        llm_context: llm.LLMContext | None,
        exposed_entities,
    ):
        """Edit file with find-and-replace."""
        write_started = False
        try:
            path_template = function_config.get("path")
            path_str = path_template.async_render(arguments, parse_result=False)
            old_text_template = function_config.get("old_text")
            old_text = old_text_template.async_render(arguments, parse_result=False)
            new_text_template = function_config.get("new_text")
            new_text = new_text_template.async_render(arguments, parse_result=False)
            allow_dirs = self._render_allow_dirs(
                hass, function_config.get("allow_dir", []), arguments
            )
            target_path = self._resolve_path(hass, path_str, allow_dirs)

            if not target_path.exists():
                return {
                    "error": "File not found",
                    "execution_status": "not_executed",
                    "retryable": True,
                }

            if not target_path.is_file():
                return {
                    "error": "Not a file",
                    "execution_status": "not_executed",
                    "retryable": True,
                }

            # Read current content
            content = await hass.async_add_executor_job(
                partial(target_path.read_text, encoding="utf-8")
            )

            # Check for text to replace
            if old_text not in content:
                return {
                    "error": "Text not found in file",
                    "execution_status": "not_executed",
                    "retryable": True,
                }

            # Check for multiple occurrences
            occurrence_count = content.count(old_text)
            if occurrence_count > 1:
                return {
                    "error": f"Text appears {occurrence_count} times in file. "
                    "A single replacement is required.",
                    "execution_status": "not_executed",
                    "retryable": True,
                }

            # Perform replacement
            new_content = content.replace(old_text, new_text, 1)

            # Write back
            write_started = True
            await hass.async_add_executor_job(
                partial(target_path.write_text, new_content, encoding="utf-8")
            )

        except EXPECTED_TOOL_ERRORS as err:
            if is_fatal_error(err):
                raise
            _LOGGER.warning("File edit failed (%s)", type(err).__name__)
            return _file_error(err, "unknown" if write_started else "not_executed")

        return {
            "success": True,
            "path": str(target_path),
            "replacements": 1,
        }
