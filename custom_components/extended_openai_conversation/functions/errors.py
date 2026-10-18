"""Safe execution errors shared by the tool boundary and compound tools."""

from __future__ import annotations

import errno
import sqlite3
from typing import Any

from aiohttp import ClientError, ClientResponseError
from httpx import RequestError
import probatio as vol
from soupsieve.util import SelectorSyntaxError
from sqlalchemy.exc import SQLAlchemyError
import yaml

from homeassistant.exceptions import HomeAssistantError, ServiceNotFound, Unauthorized
from homeassistant.helpers import llm

from ..exceptions import (
    CallServiceError,
    EntityNotExposed,
    EntityNotFound,
    FunctionLoadFailed,
    FunctionNotFound,
    InvalidFunction,
    NativeNotFound,
)

EXPECTED_TOOL_ERRORS = (
    HomeAssistantError,
    vol.Invalid,
    KeyError,
    TypeError,
    ValueError,
    IndexError,
    OSError,
    RuntimeError,
    sqlite3.Error,
    SQLAlchemyError,
    ClientError,
    RequestError,
    yaml.YAMLError,
    SelectorSyntaxError,
)

UNCONFIRMED_OPERATION_MESSAGE = (
    "The operation may already have executed, in whole or in part. "
    "Do not retry it directly. Query the actual state first."
)


def is_fatal_error(error: Exception) -> bool:
    """Configuration, programming and resource failures cannot be fixed by an LLM."""
    return (
        isinstance(
            error,
            (
                MemoryError,
                RecursionError,
                ImportError,
                AssertionError,
                FunctionLoadFailed,
                InvalidFunction,
                NativeNotFound,
                SelectorSyntaxError,
            ),
        )
        or (
            isinstance(error, OSError)
            and error.errno
            in {errno.ENOMEM, errno.ENOSPC, errno.EMFILE, errno.ENFILE, errno.EIO}
        )
        or (
            isinstance(error, sqlite3.Error)
            and (getattr(error, "sqlite_errorcode", 0) & 0xFF)
            in {
                sqlite3.SQLITE_FULL,
                sqlite3.SQLITE_NOMEM,
                sqlite3.SQLITE_IOERR,
                sqlite3.SQLITE_CANTOPEN,
                sqlite3.SQLITE_CORRUPT,
                sqlite3.SQLITE_NOTADB,
            }
        )
    )


def is_permission_error(error: Exception) -> bool:
    """Never suggest changing parameters to evade an authorization decision."""
    return (
        isinstance(error, (PermissionError, EntityNotExposed, Unauthorized))
        or (isinstance(error, ClientResponseError) and error.status in {401, 403})
        or (
            isinstance(error, sqlite3.Error)
            and (getattr(error, "sqlite_errorcode", 0) & 0xFF)
            in {sqlite3.SQLITE_AUTH, sqlite3.SQLITE_PERM, sqlite3.SQLITE_READONLY}
        )
    )


def execution_error(
    error: Exception,
    *,
    status: str = "unknown",
    retryable: bool | None = None,
) -> llm.ToolResult:
    """Disclose error types and fixed descriptions, never exception values/causes."""
    if is_permission_error(error):
        message = "Access denied by the configured permissions."
        retryable = False
    elif isinstance(error, EntityNotFound):
        message = "A requested entity does not exist."
    elif isinstance(error, ServiceNotFound):
        message = "The requested Home Assistant service does not exist."
    elif isinstance(error, FunctionNotFound):
        message = "The requested tool is not available."
    elif isinstance(error, CallServiceError):
        message = "The service requires an explicit target."
    elif isinstance(error, (vol.Invalid, KeyError, TypeError, ValueError, IndexError)):
        message = "The supplied parameters are missing, invalid or incompatible."
    elif isinstance(error, TimeoutError):
        message = "The operation did not finish within the waiting period."
    elif isinstance(error, sqlite3.Error):
        message = "The database operation failed."
    elif isinstance(error, (ClientError, RequestError, ConnectionError)):
        message = "The remote request failed."
    else:
        message = "The tool operation failed."
    data: dict[str, Any] = {
        "error": type(error).__name__,
        "error_text": message,
        "execution_status": status,
        "retryable": (
            status in {"not_executed", "failed"} if retryable is None else retryable
        ),
    }
    if status in {"unknown", "partial"}:
        data["requires_state_check"] = True
        data["retryable"] = False
        data["recovery_action"] = UNCONFIRMED_OPERATION_MESSAGE
    if is_permission_error(error):
        data["permission_denied"] = True
    if isinstance(error, ClientResponseError):
        data["http_status"] = error.status
    return llm.ToolResult(data=data, error=True)


def result_is_error(result: Any, config: dict[str, Any]) -> bool:
    """Read explicit status, or legacy envelopes only from their known producers."""
    if isinstance(result, llm.ToolResult):
        return result.error
    if not isinstance(result, dict):
        return False
    function_type = config.get("type")
    if function_type == "bash":
        return "error" in result or result.get("exit_code", 0) != 0
    if function_type in {"read_file", "write_file", "edit_file"} or (
        function_type == "native" and config.get("name") == "execute_service_single"
    ):
        return "error" in result or result.get("success") is False
    # SQLite rows, parsed templates, scripts and remote responses are business
    # data. Their field names do not define execution status. Composite failures
    # and native service batches already return explicit ToolResult instances.
    return False


def result_blocks_replay(result: llm.ToolResult) -> bool:
    """An unconfirmed operation or permission denial is never a safe retry."""
    return bool(result.data.get("permission_denied")) or result.data.get(
        "execution_status"
    ) in {"unknown", "partial", "scheduled"}


def function_is_read_only(config: dict[str, Any]) -> bool:
    """Only claim retry safety for known read operations, not arbitrary scripts."""
    function_type = config.get("type")
    if function_type in {"sqlite", "read_file", "template"}:
        return True
    if function_type == "native":
        return bool(config.get("name", "").startswith("get_"))
    if function_type in {"rest", "scrape"}:
        return config.get("method", "GET").upper() in {"GET", "HEAD", "OPTIONS"}
    return False
