"""Base entity for Extended OpenAI Conversation."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncGenerator
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
from mimetypes import guess_file_type
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from openai import AsyncClient, AsyncStream
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
import orjson
import probatio as vol

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import device_registry as dr, llm, template
from homeassistant.helpers.entity import Entity
from homeassistant.util import slugify

from .const import (
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_EXTRA_BODY,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_REASONING_EFFORT,
    CONF_SERVICE_TIER,
    CONF_SHORTEN_TOOL_CALL_ID,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_EXTRA_BODY,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SERVICE_TIER,
    DEFAULT_SHORTEN_TOOL_CALL_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
)
from .exceptions import (
    FunctionNotFound,
    InvalidFunction,
    ParseArgumentsFailed,
    TokenLengthExceededError,
)
from .functions import get_function
from .functions.errors import (
    EXPECTED_TOOL_ERRORS,
    execution_error,
    function_is_read_only,
    is_fatal_error,
    result_blocks_replay,
    result_is_error,
)
from .helpers import get_model_config

if TYPE_CHECKING:
    from . import ExtendedOpenAIConfigEntry

_LOGGER = logging.getLogger(__name__)

# Internal tool-round limit, including malformed calls. Allow a final text reply.
MAX_TOOL_ITERATIONS = 20

_NATIVE_TOOL_CALL_EXTRA_CONTENT_KEY = "tool_call_extra_content_by_id"

_TOOL_ARGUMENTS_SCHEMA = vol.Schema(dict)


@dataclass
class _ToolCallResponse:
    """Request-local wire data; malformed arguments must never become HA ToolInput."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list, repr=False)
    errors: dict[int, dict[str, str]] = field(default_factory=dict, repr=False)
    content: str = field(default="", repr=False)
    complete: bool = False
    has_tool_calls: bool = False


def _tool_call_key(tool_input: llm.ToolInput) -> tuple[str, str]:
    """Identify a completed call when a correction changes its provider ID."""
    return tool_input.tool_name, json.dumps(tool_input.tool_args, sort_keys=True)


def _shorten_tool_call_id(tool_call_id: str) -> str:
    """Shorten tool call ID to exactly 9 alphanumeric characters as Mistral requires."""
    import hashlib

    return hashlib.sha256(tool_call_id.encode()).hexdigest()[:9]


def _tool_error_messages(
    response: _ToolCallResponse, shorten_tool_call_id: bool
) -> list[ChatCompletionMessageParam]:
    """Pair the original wire calls with errors, without inventing HA arguments.

    Chat Completions allows arguments to be an unparsed string. HA requires a
    dict, so these pairs exist only in the retry request, never in ChatLog.
    A missing identity, duplicate ID or unfinished response cannot be paired:
    return no feedback and retry from the last complete history instead.
    """
    if not response.complete or not response.tool_calls or not response.errors:
        return []
    calls: list[Any] = []
    results: list[ChatCompletionMessageParam] = []
    seen_ids: set[str] = set()
    for index, call in enumerate(response.tool_calls):
        if not call["id"] or not call["name"]:
            return []
        call_id = (
            _shorten_tool_call_id(call["id"]) if shorten_tool_call_id else call["id"]
        )
        if call_id in seen_ids:
            return []
        seen_ids.add(call_id)
        wire_call = {
            "id": call_id,
            "type": "function",
            "function": {"name": call["name"], "arguments": call["arguments"]},
        }
        if "extra_content" in call:
            wire_call["extra_content"] = call["extra_content"]
        calls.append(wire_call)
        results.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(
                    response.errors.get(index, {"status": "not_executed"})
                ),
            }
        )
    return [
        {"role": "assistant", "content": response.content or None, "tool_calls": calls},
        *results,
    ]


def _extract_extra_content(value: Any) -> dict[str, Any] | None:
    """Extract opaque provider-specific extra_content from an SDK response model."""
    extra_content = getattr(value, "extra_content", None)
    if isinstance(extra_content, dict):
        return extra_content

    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, dict):
        extra_content = model_extra.get("extra_content")
        if isinstance(extra_content, dict):
            return extra_content

    return None


def _get_tool_call_extra_content(
    native: Any, tool_call_id: str
) -> dict[str, Any] | None:
    """Return provider metadata stored in an AssistantContent native payload."""
    if not isinstance(native, dict):
        return None

    extra_content_by_id = native.get(_NATIVE_TOOL_CALL_EXTRA_CONTENT_KEY)
    if not isinstance(extra_content_by_id, dict):
        return None

    extra_content = extra_content_by_id.get(tool_call_id)
    return extra_content if isinstance(extra_content, dict) else None


def _adjust_schema(schema: dict[str, Any]) -> None:
    """Adjust the schema to be compatible with OpenAI API."""
    if schema["type"] == "object":
        schema.setdefault("strict", True)
        schema.setdefault("additionalProperties", False)
        if "properties" not in schema:
            return

        if "required" not in schema:
            schema["required"] = []

        # Ensure all properties are required
        for prop, prop_info in schema["properties"].items():
            _adjust_schema(prop_info)
            if prop not in schema["required"]:
                prop_info["type"] = [prop_info["type"], "null"]
                schema["required"].append(prop)

    elif schema["type"] == "array":
        if "items" not in schema:
            return

        _adjust_schema(schema["items"])


def _format_structured_output(
    schema: vol.Schema, llm_api: llm.APIInstance | None
) -> dict[str, Any]:
    """Format the schema to be compatible with OpenAI API."""
    result: dict[str, Any] = vol.to_openapi(
        schema,
        custom_serializer=(
            llm_api.custom_serializer if llm_api else llm.selector_serializer
        ),
    )

    _adjust_schema(result)

    return result


def _make_tool_result_content(
    agent_id: str,
    tool_call_id: str,
    tool_name: str,
    data: dict[str, Any],
    *,
    error: bool = False,
) -> conversation.ToolResultContent:
    """Build a ToolResultContent across HA versions.

    HA 2026.10 replaced the `tool_result` dict field with `result`, an
    `llm.ToolResult` wrapper; older versions only accept `tool_result`.
    """
    tool_result_cls = getattr(llm, "ToolResult", None)
    payload: dict[str, Any] = (
        {"result": tool_result_cls(data=data, error=error)}
        if tool_result_cls is not None
        else {"tool_result": data}
    )
    return conversation.ToolResultContent(
        agent_id=agent_id, tool_call_id=tool_call_id, tool_name=tool_name, **payload
    )


def _tool_result_data(content: conversation.ToolResultContent) -> Any:
    """Return a tool result's data, avoiding HA 2026.10's deprecated accessor."""
    result = getattr(content, "result", None)
    if result is not None:
        return result.data
    return content.tool_result


def _tool_result(content: conversation.ToolResultContent) -> llm.ToolResult:
    """Read the native result, retaining error flags when replaying it."""
    result = getattr(content, "result", None)
    if result is not None:
        return cast(llm.ToolResult, result)
    return llm.ToolResult(data=content.tool_result)


def _render_extra_body(
    extra_body_raw: str, hass: HomeAssistant, model: str
) -> dict[str, Any] | None:
    """Render the extra_body template option to a dict, or None if unset/invalid.

    extra_body_raw is a Jinja-templatable JSON string (e.g. from the
    CONF_EXTRA_BODY option); an empty/whitespace-only string means the
    feature is disabled. A template or JSON error is logged as a warning
    and treated as unset, rather than failing the conversation turn.
    """
    if not extra_body_raw.strip():
        return None

    try:
        rendered = template.Template(extra_body_raw, hass).async_render(
            parse_result=False
        )
    except TemplateError as err:
        _LOGGER.warning(
            "Invalid extra_body template for %s, ignoring (%s)",
            model,
            type(err).__name__,
        )
        return None

    if not rendered.strip():
        return None

    try:
        parsed: dict[str, Any] = json.loads(rendered)
        return parsed
    except json.JSONDecodeError as err:
        _LOGGER.warning(
            "Invalid extra_body JSON for %s, ignoring (%s)", model, type(err).__name__
        )
        return None


def encode_attachments(
    chat_content: list[conversation.Content],
) -> dict[int, list[ChatCompletionContentPartImageParam]]:
    """Read and base64-encode attachments. BLOCKING - run in an executor.

    Returns a map of {index in chat_content: list of image content parts}.
    """
    encoded: dict[int, list[ChatCompletionContentPartImageParam]] = {}

    for index, content in enumerate(chat_content):
        attachments = getattr(content, "attachments", None)
        if content.role != "user" or not attachments:
            continue

        parts: list[ChatCompletionContentPartImageParam] = []
        for attachment in attachments:
            file_path = Path(attachment.path)
            if not file_path.exists():
                raise HomeAssistantError(
                    f"Attachment does not exist: {file_path.as_posix()}"
                )

            mime_type = attachment.mime_type or guess_file_type(file_path)[0]
            if not mime_type or not mime_type.startswith("image/"):
                raise HomeAssistantError(
                    f"Unsupported attachment type {mime_type or 'unknown'} "
                    f"for {file_path.as_posix()}"
                )
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"

            b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
            parts.append(
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url={"url": f"data:{mime_type};base64,{b64}"},
                )
            )

        if parts:
            encoded[index] = parts

    return encoded


def _convert_content_to_param(
    chat_content: list[conversation.Content],
    shorten_tool_call_id: bool = False,
    attachment_parts: dict[int, list[ChatCompletionContentPartImageParam]]
    | None = None,
) -> list[ChatCompletionMessageParam]:
    """Convert chat log content to OpenAI message format."""
    messages: list[ChatCompletionMessageParam] = []
    attachment_parts = attachment_parts or {}

    for index, content in enumerate(chat_content):
        if content.role == "system":
            messages.append({"role": "system", "content": content.content})
        elif content.role == "user":
            if (parts := attachment_parts.get(index)) is None:
                messages.append({"role": "user", "content": content.content})
            else:
                # Multipart form: text first, then one image_url block per
                # attachment.
                multipart: list[
                    ChatCompletionContentPartTextParam
                    | ChatCompletionContentPartImageParam
                ] = []
                if content.content:
                    multipart.append(
                        ChatCompletionContentPartTextParam(
                            type="text", text=content.content
                        )
                    )
                multipart.extend(parts)
                user_msg: ChatCompletionUserMessageParam = {
                    "role": "user",
                    "content": multipart,
                }
                messages.append(user_msg)
        elif content.role == "assistant":
            msg: ChatCompletionAssistantMessageParam = {"role": "assistant"}
            if content.content:
                msg["content"] = content.content
            if content.tool_calls:
                serialized_tool_calls: list[Any] = []
                for tool_call in content.tool_calls:
                    serialized_tool_call: dict[str, Any] = {
                        "id": (
                            _shorten_tool_call_id(tool_call.id)
                            if shorten_tool_call_id
                            else tool_call.id
                        ),
                        "type": "function",
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(tool_call.tool_args),
                        },
                    }

                    extra_content = _get_tool_call_extra_content(
                        content.native, tool_call.id
                    )
                    if extra_content is not None:
                        serialized_tool_call["extra_content"] = extra_content

                    serialized_tool_calls.append(serialized_tool_call)

                msg["tool_calls"] = serialized_tool_calls
            # Some OpenAI-compatible APIs (like Mistral) reject empty tool_calls arrays
            # Remove tool_calls field if it's an empty array to maintain compatibility
            if msg.get("tool_calls") == []:
                msg.pop("tool_calls", None)
            # Some OpenAI-compatible APIs (vLLM, SGLang, etc.) reject assistant
            # messages missing the "content" key entirely; ensure it's always
            # present, matching the Mistral empty-tool_calls guard above
            if "content" not in msg:
                msg["content"] = None
            messages.append(msg)
        elif content.role == "tool_result":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _shorten_tool_call_id(content.tool_call_id)
                    if shorten_tool_call_id
                    else content.tool_call_id,
                    "content": orjson.dumps(_tool_result_data(content)).decode(),
                }
            )

    return messages


class ExtendedOpenAIBaseLLMEntity(Entity):
    """Extended OpenAI base entity."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self, entry: ExtendedOpenAIConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="OpenAI",
            model=subentry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def _client(self) -> AsyncClient:
        """Return the OpenAI client."""
        return self.entry.runtime_data

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        function_tools: list[dict[str, Any]],
        exposed_entities: list[dict[str, Any]],
        llm_context: llm.LLMContext | None = None,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
    ) -> None:
        """Generate an answer for the chat log with streaming support."""
        options = self.subentry.data
        model = options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        max_function_calls = options.get(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        )
        shorten_tool_call_id = options.get(
            CONF_SHORTEN_TOOL_CALL_ID,
            DEFAULT_SHORTEN_TOOL_CALL_ID,
        )

        # Get model-specific configuration
        model_config = get_model_config(model)

        # Encoding reads attachment files from disk - run in an executor, and
        # do it once up front since chat_log.content only grows (attachments
        # already present keep the same index across the tool-calling loop
        # below).
        attachment_parts = await self.hass.async_add_executor_job(
            encode_attachments, chat_log.content
        )

        messages = _convert_content_to_param(
            chat_log.content, shorten_tool_call_id, attachment_parts
        )

        # Build functions list from custom functions
        tools: list[ChatCompletionToolParam] = [
            ChatCompletionToolParam(
                type="function",
                function=func_spec["spec"],
            )
            for func_spec in function_tools
        ]

        # Build API parameters based on model configuration
        api_kwargs: dict[str, Any] = {
            "model": model,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # Add token limit parameter based on model support
        max_tokens = options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        if model_config["supports_max_completion_tokens"]:
            api_kwargs["max_completion_tokens"] = max_tokens
        elif model_config["supports_max_tokens"]:
            api_kwargs["max_tokens"] = max_tokens

        # Add top_p if supported
        if model_config["supports_top_p"]:
            api_kwargs["top_p"] = options.get(CONF_TOP_P, DEFAULT_TOP_P)

        # Add temperature if supported
        if model_config["supports_temperature"]:
            api_kwargs["temperature"] = options.get(
                CONF_TEMPERATURE, DEFAULT_TEMPERATURE
            )

        # Add reasoning_effort if supported (o1, o3, o4, gpt-5 models)
        if model_config.get("supports_reasoning_effort"):
            api_kwargs["reasoning_effort"] = options.get(
                CONF_REASONING_EFFORT, DEFAULT_REASONING_EFFORT
            )

        # Add service_tier if supported (o3, o4, gpt-5 models)
        if model_config.get("supports_service_tier"):
            api_kwargs["service_tier"] = options.get(
                CONF_SERVICE_TIER, DEFAULT_SERVICE_TIER
            )

        # Add extra_body if configured - passthrough for OpenAI-compatible
        # backends that accept extra request-body fields (ollama, llama.cpp,
        # vLLM, LM Studio, etc.). E.g. {"chat_template_kwargs":
        # {"enable_thinking": false}} to disable Qwen3 reasoning, or
        # {"cache_prompt": true} for llama.cpp prompt caching.
        extra_body = _render_extra_body(
            options.get(CONF_EXTRA_BODY, DEFAULT_EXTRA_BODY) or "", self.hass, model
        )
        if extra_body is not None:
            api_kwargs["extra_body"] = extra_body

        # Add structured output format if provided
        if structure is not None:
            api_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": slugify(structure_name),
                    "strict": True,
                    "schema": _format_structured_output(structure, chat_log.llm_api),
                },
            }

        # Add tools if available
        tool_kwargs: dict[str, Any] = {}
        if tools:
            tool_kwargs["tools"] = tools
            tool_kwargs["tool_choice"] = "auto"

        # Valid and invalid tool-call responses share the configured budget.
        # Even an unlimited configuration remains bounded by the internal cap.
        tool_round_limit = (
            MAX_TOOL_ITERATIONS
            if max_function_calls < 0
            else min(max_function_calls, MAX_TOOL_ITERATIONS)
        )
        tool_rounds = 0
        completed_results: dict[tuple[str, str], conversation.ToolResultContent] = {}
        completed_ids: dict[str, tuple[str, str]] = {}
        results_by_id: dict[str, conversation.ToolResultContent] = {}
        blocked_functions: dict[str, conversation.ToolResultContent] = {}
        retrying = False
        replay_results: dict[tuple[str, str], conversation.ToolResultContent] = {}
        request_messages = messages

        # Reserve one response for final text after the last successful tool round.
        for n_requests in range(MAX_TOOL_ITERATIONS + 1):
            tools_allowed = bool(tools) and tool_rounds < tool_round_limit
            if tools:
                tool_kwargs["tool_choice"] = "auto" if tools_allowed else "none"
            history_before_request = chat_log.content.copy()
            _LOGGER.debug(
                "Requesting response (iteration=%s, correction=%s, messages=%s)",
                n_requests + 1,
                retrying,
                len(request_messages),
            )
            stream = await self._client.chat.completions.create(
                messages=request_messages,
                **api_kwargs,
                **tool_kwargs,
            )
            pending_tool_calls: list[llm.ToolInput] = []
            response = _ToolCallResponse()
            response_committed = False
            try:
                try:
                    deltas = (
                        self._validated_retry_stream(
                            chat_log, stream, completed_ids, response
                        )
                        if tools
                        else self._transform_stream(chat_log, stream, response)
                    )
                    async for content in chat_log.async_add_delta_content_stream(
                        self.entity_id, deltas
                    ):
                        if (
                            isinstance(content, conversation.AssistantContent)
                            and content.tool_calls
                        ):
                            pending_tool_calls.extend(content.tool_calls)
                    if response.has_tool_calls and not tools_allowed:
                        # Some compatible backends ignore tool_choice="none".
                        # Do not commit or execute calls beyond the shared limit.
                        chat_log.content[:] = history_before_request
                        raise HomeAssistantError("Tool call limit reached")
                finally:
                    # Count a response once, even for multiple calls, invalid
                    # JSON, missing identities or an unfinished argument stream.
                    if response.has_tool_calls:
                        tool_rounds += 1
                    # The validator closes tool-enabled streams before publishing.
                    if not tools:
                        await stream.close()
                response_committed = True
            except ParseArgumentsFailed as err:
                # HA may have streamed text, but no tool in this response has
                # run. Restore only this request's checkpoint, never the turn.
                chat_log.content[:] = history_before_request
                _LOGGER.warning(
                    "Invalid tool arguments (iteration=%s, tool_rounds=%s, "
                    "tool_limit=%s, reason=%s, tool_index=%s, arguments_length=%s, "
                    "position=%s, finish_reason=%s)",
                    n_requests + 1,
                    tool_rounds,
                    tool_round_limit,
                    err.reason,
                    err.tool_index,
                    err.arguments_length,
                    err.position,
                    err.finish_reason,
                )
                if (
                    not tools_allowed
                    or not response.has_tool_calls
                    or tool_rounds >= tool_round_limit
                ):
                    raise
                retrying = True
                replay_results = completed_results.copy()
                # Use only actual parser/validator output. If the wire call is
                # incomplete, replay the clean checkpoint without prompts or
                # fabricated tool IDs/results. Both paths use the shared budget.
                request_messages = [
                    *messages,
                    *_tool_error_messages(response, shorten_tool_call_id),
                ]
                continue
            finally:
                if not response_committed:
                    chat_log.content[:] = history_before_request

            if retrying:
                _LOGGER.debug("Tool argument correction succeeded")

            if pending_tool_calls:
                _LOGGER.debug(
                    "Response includes %s tool calls", len(pending_tool_calls)
                )

            # A new model response may not rewrite an unconfirmed operation to
            # evade replay protection. Calls already in this batch are independent.
            blocked_before_batch = blocked_functions.copy()
            answered: set[str] = set()
            active_tool: llm.ToolInput | None = None
            batch_has_error = False
            try:
                for tool_input in pending_tool_calls:
                    if self.hass.is_stopping:
                        raise HomeAssistantError("Home Assistant is stopping")
                    call_key = _tool_call_key(tool_input)
                    function_tool = next(
                        (
                            f
                            for f in function_tools
                            if f["spec"]["name"] == tool_input.tool_name
                        ),
                        None,
                    )
                    previous_result = results_by_id.get(tool_input.id)
                    cached = completed_results.get(call_key)
                    if (
                        previous_result is None
                        and cached is not None
                        and (
                            call_key in replay_results
                            or result_blocks_replay(_tool_result(cached))
                        )
                    ):
                        previous_result = cached
                    if previous_result is not None:
                        outcome = _tool_result(previous_result)
                        _LOGGER.debug("Reusing recorded tool result")
                    elif tool_input.tool_name in blocked_before_batch:
                        previous = _tool_result(
                            blocked_before_batch[tool_input.tool_name]
                        )
                        outcome = llm.ToolResult(
                            data={
                                "error": "UnsafeRetryBlocked",
                                "execution_status": "not_executed",
                                "retryable": False,
                                "previous_execution_status": previous.data.get(
                                    "execution_status"
                                ),
                                "requires_state_check": bool(
                                    previous.data.get("requires_state_check")
                                ),
                                "permission_denied": bool(
                                    previous.data.get("permission_denied")
                                ),
                            },
                            error=True,
                        )
                    elif function_tool is None:
                        outcome = execution_error(
                            FunctionNotFound(tool_input.tool_name),
                            status="not_executed",
                        )
                    else:
                        active_tool = tool_input
                        try:
                            content = await self._execute_function_tool(
                                function_tool, tool_input, llm_context, exposed_entities
                            )
                            outcome = _tool_result(content)
                        except EXPECTED_TOOL_ERRORS as err:
                            if is_fatal_error(err) or self.hass.is_stopping:
                                raise
                            outcome = execution_error(err)
                            _LOGGER.warning(
                                "Tool execution failed (%s)", type(err).__name__
                            )
                    tool_result_content = _make_tool_result_content(
                        self.entity_id,
                        tool_input.id,
                        tool_input.tool_name,
                        outcome.data,
                        error=outcome.error,
                    )
                    # Record before notifying listeners: an observer failure must
                    # never result in a second side effect or a duplicate result.
                    answered.add(tool_input.id)
                    completed_ids[tool_input.id] = call_key
                    results_by_id[tool_input.id] = tool_result_content
                    if not outcome.error or result_blocks_replay(outcome):
                        completed_results[call_key] = tool_result_content
                    if outcome.data.get("permission_denied") or outcome.data.get(
                        "execution_status"
                    ) in {"unknown", "partial"}:
                        blocked_functions[tool_input.tool_name] = tool_result_content
                    batch_has_error |= outcome.error
                    self._add_tool_result(chat_log, tool_result_content)
                    active_tool = None
            finally:
                # No BaseException is swallowed. In-flight cancellation/fatal
                # failures remain unknown; later calls were never executed.
                for tool_input in pending_tool_calls:
                    if tool_input.id not in answered:
                        in_flight = (
                            active_tool is not None and active_tool.id == tool_input.id
                        )
                        self._add_tool_result(
                            chat_log,
                            _make_tool_result_content(
                                self.entity_id,
                                tool_input.id,
                                tool_input.tool_name,
                                {
                                    "error": "ExecutionInterrupted"
                                    if in_flight
                                    else "NotExecuted",
                                    "execution_status": "unknown"
                                    if in_flight
                                    else "not_executed",
                                    "retryable": False,
                                    "requires_state_check": in_flight,
                                },
                                error=True,
                            ),
                        )

            # Update messages for next iteration
            messages = _convert_content_to_param(
                chat_log.content, shorten_tool_call_id, attachment_parts
            )

            # Check if we need to continue (if there are pending tool results)
            if not chat_log.unresponded_tool_results:
                break
            retrying = batch_has_error
            replay_results = completed_results.copy() if retrying else {}
            request_messages = messages

    def _add_tool_result(
        self, chat_log: conversation.ChatLog, content: conversation.ToolResultContent
    ) -> None:
        """Publish external results through HA's API and its streaming listener."""
        before = len(chat_log.content)
        try:
            chat_log.async_add_assistant_content_without_tools(content)
        except Exception as err:
            if len(chat_log.content) == before or is_fatal_error(err):
                raise
            _LOGGER.warning("Tool result observer failed (%s)", type(err).__name__)
        if chat_log.delta_listener:
            try:
                chat_log.delta_listener(chat_log, content.as_dict())
            except Exception as err:
                if is_fatal_error(err):
                    raise
                _LOGGER.warning("Tool result listener failed (%s)", type(err).__name__)

    async def _validated_retry_stream(
        self,
        chat_log: conversation.ChatLog,
        stream: AsyncStream[ChatCompletionChunk],
        completed_ids: dict[str, tuple[str, str]],
        response: _ToolCallResponse,
    ) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
        """Reuse atomic validation before any tool-enabled response reaches Assist."""
        try:
            deltas = [
                delta
                async for delta in self._transform_stream(chat_log, stream, response)
            ]
        finally:
            await stream.close()
        tool_calls = [
            tool_call
            for delta in deltas
            for tool_call in delta.get("tool_calls", []) or []
        ]
        if not tool_calls and not response.content:
            raise ParseArgumentsFailed("", reason="empty_response")
        for tool_call in tool_calls:
            completed_key = completed_ids.get(tool_call.id)
            if completed_key is not None and completed_key != _tool_call_key(tool_call):
                raise ParseArgumentsFailed("", reason="invalid_tool_call_id")
        # A preamble must not announce success before tools run. Failed streams
        # published no role/content/native, so each valid response starts once.
        for delta in deltas:
            if not tool_calls or "content" not in delta:
                yield delta

    async def _transform_stream(
        self,
        chat_log: conversation.ChatLog,
        result: AsyncStream[ChatCompletionChunk],
        response: _ToolCallResponse | None = None,
    ) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
        """Transform OpenAI stream to Home Assistant format."""
        if response is None:
            response = _ToolCallResponse()
        current_tool_calls: dict[int, dict[str, Any]] = {}
        # Collect provider metadata across the whole response and attach it
        # once. HA only allows AssistantContent.native to be set a single time
        # per message; yielding native on every tool_calls finish_reason would
        # raise RuntimeError on loosely compatible servers that emit that
        # finish reason more than once.
        tool_call_extra_content_by_id: dict[str, dict[str, Any]] = {}
        first_chunk = True
        finish_reason = None
        failure: ParseArgumentsFailed | None = None
        seen_ids: set[str] = set()

        async for chunk in result:
            _LOGGER.debug("Received response chunk (choices=%s)", len(chunk.choices))

            # Signal new assistant message on first chunk
            if first_chunk:
                yield {"role": "assistant"}
                first_chunk = False

            if not chunk.choices:
                # Track usage from final chunk if available
                if chunk.usage:
                    chat_log.async_trace(
                        {
                            "stats": {
                                "input_tokens": chunk.usage.prompt_tokens,
                                "output_tokens": chunk.usage.completion_tokens,
                            }
                        }
                    )
                    if chunk.usage.total_tokens > self.subentry.data.get(
                        CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD
                    ):
                        await self._truncate_message_history(chat_log)
                continue

            choice = chunk.choices[0]
            finish_reason = choice.finish_reason
            delta = choice.delta

            if delta.content:
                # Ensure content is a string (Mistral might return unexpected types)
                content_value = delta.content
                if not isinstance(content_value, str):
                    _LOGGER.warning(
                        "Received non-string content from API (type: %s)",
                        type(content_value),
                    )
                    content_value = str(content_value) if content_value else ""
                if content_value:
                    response.content += content_value
                    yield {"content": content_value}

            if delta.tool_calls:
                response.has_tool_calls = True
                for tool_call_delta in delta.tool_calls:
                    idx = tool_call_delta.index
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {
                            "id": tool_call_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }

                    if tool_call_delta.id:
                        current_tool_calls[idx]["id"] = tool_call_delta.id

                    extra_content = _extract_extra_content(tool_call_delta)
                    if extra_content is not None:
                        current_tool_calls[idx]["extra_content"] = extra_content

                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            current_tool_calls[idx]["name"] = (
                                tool_call_delta.function.name
                            )
                        if tool_call_delta.function.arguments:
                            current_tool_calls[idx]["arguments"] += (
                                tool_call_delta.function.arguments
                            )

            if current_tool_calls and (choice.finish_reason in {"tool_calls", "stop"}):
                # Yield all accumulated tool calls (marked as external since we handle them ourselves)
                tool_calls_list = []
                for idx in sorted(current_tool_calls.keys()):
                    tool_call = current_tool_calls[idx]
                    call_index = len(response.tool_calls)
                    response.tool_calls.append(tool_call.copy())
                    if (
                        not tool_call["id"]
                        or not tool_call["name"]
                        or tool_call["id"] in seen_ids
                    ):
                        failure = failure or ParseArgumentsFailed(
                            tool_call["arguments"],
                            tool_index=idx,
                            reason="invalid_tool_call_id",
                            finish_reason=choice.finish_reason,
                        )
                    seen_ids.add(tool_call["id"])
                    try:
                        args = _TOOL_ARGUMENTS_SCHEMA(
                            json.loads(tool_call["arguments"])
                        )
                    except (json.JSONDecodeError, vol.Invalid) as err:
                        response.errors[call_index] = {
                            "error": type(err).__name__,
                            "error_text": str(err),
                        }
                        failure = failure or ParseArgumentsFailed(
                            tool_call["arguments"],
                            tool_index=idx,
                            position=getattr(err, "pos", None),
                            reason=(
                                "invalid_json"
                                if isinstance(err, json.JSONDecodeError)
                                else "invalid_arguments"
                            ),
                            finish_reason=choice.finish_reason,
                        )
                        continue

                    if failure is not None:
                        continue

                    extra_content = tool_call.get("extra_content")
                    if tool_call["id"] and extra_content is not None:
                        tool_call_extra_content_by_id[tool_call["id"]] = extra_content

                    tool_calls_list.append(
                        llm.ToolInput(
                            id=tool_call["id"],
                            tool_name=tool_call["name"],
                            tool_args=args,
                            external=True,  # Mark as external so ChatLog doesn't try to execute
                        )
                    )
                if tool_calls_list and failure is None:
                    yield {"tool_calls": tool_calls_list}
                current_tool_calls.clear()
            if choice.finish_reason == "length":
                raise TokenLengthExceededError(
                    self.subentry.data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
                )

            if choice.finish_reason == "stop":
                break

        if current_tool_calls:
            if finish_reason is not None:
                raise HomeAssistantError("Tool call response did not finish normally")
            idx = min(current_tool_calls)
            failure = failure or ParseArgumentsFailed(
                current_tool_calls[idx]["arguments"],
                tool_index=idx,
                reason="unfinished_stream",
                finish_reason="eof",
            )

        response.complete = (
            finish_reason in {"tool_calls", "stop"} and not current_tool_calls
        )
        if failure is not None:
            # Finish collecting this response before reporting a failed batch.
            # Neither HA nor our executor has executed any of its calls.
            raise failure from None

        if not first_chunk and tool_call_extra_content_by_id:
            yield {
                "native": {
                    _NATIVE_TOOL_CALL_EXTRA_CONTENT_KEY: tool_call_extra_content_by_id
                }
            }

    async def _execute_function_tool(
        self,
        function_tool: dict[str, Any],
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> conversation.ToolResultContent:
        """Execute a custom function."""
        arguments: dict[str, Any] = deepcopy(tool_input.tool_args)
        function_config = function_tool["function"]
        try:
            function = get_function(function_config["type"])
        except (KeyError, FunctionNotFound) as err:
            raise InvalidFunction("configured tool") from err
        missing = [
            name
            for name in function_tool["spec"].get("parameters", {}).get("required", [])
            if name not in arguments
        ]
        if missing:
            outcome = execution_error(KeyError(), status="not_executed")
            outcome.data["missing_parameters"] = missing
        elif self.should_run_in_background(arguments):
            # create a delayed function and execute in background
            function_config = self.get_delayed_function_config(
                function_config, arguments
            )
            try:
                # Only the new delay is unvalidated. Revalidating the original
                # function would reject templates HA has already compiled.
                function_config["sequence"][0] = get_function("script").validate_schema(
                    function_config["sequence"][0]
                )
            except InvalidFunction as err:
                cause = err.__cause__
                outcome = execution_error(
                    cause if isinstance(cause, Exception) else ValueError(),
                    status="not_executed",
                )
            else:
                coroutine = self._async_run_background_tool(
                    function_config, arguments, llm_context, exposed_entities
                )
                try:
                    self.entry.async_create_task(self.hass, coroutine)
                except Exception:
                    coroutine.close()
                    raise
                outcome = llm.ToolResult(
                    data={
                        "result": "Scheduled",
                        "execution_status": "scheduled",
                        "completion_confirmed": False,
                        "retryable": False,
                    }
                )
        else:
            try:
                result = await function.execute(
                    self.hass, function_config, arguments, llm_context, exposed_entities
                )
            except EXPECTED_TOOL_ERRORS as err:
                if is_fatal_error(err) or self.hass.is_stopping:
                    raise
                outcome = execution_error(
                    err,
                    status="failed"
                    if function_is_read_only(function_config)
                    else "unknown",
                )
                _LOGGER.warning("Tool execution failed (%s)", type(err).__name__)
            else:
                try:
                    if isinstance(result, llm.ToolResult):
                        outcome = result
                    elif result_is_error(result, function_config):
                        data = (
                            result if isinstance(result, dict) else {"results": result}
                        )
                        # Preserve legacy error fields; add only execution metadata.
                        outcome = llm.ToolResult(data=dict(data), error=True)
                    else:
                        outcome = llm.ToolResult(data={"result": str(result)})
                    if outcome.error and (
                        "execution_status" not in outcome.data
                        or outcome.data.get("execution_status")
                        in {"unknown", "partial"}
                    ):
                        data = dict(outcome.data)
                        read_only = function_is_read_only(function_config)
                        data.setdefault(
                            "execution_status", "failed" if read_only else "unknown"
                        )
                        data.setdefault("retryable", read_only)
                        if data["execution_status"] in {"unknown", "partial"}:
                            data["retryable"] = False
                            data.setdefault("requires_state_check", True)
                        outcome = llm.ToolResult(data=data, error=True)
                    orjson.dumps(outcome.data)
                except Exception as err:
                    if is_fatal_error(err):
                        raise
                    outcome = llm.ToolResult(
                        data={
                            "error": "ResultSerializationError",
                            "execution_status": "unknown",
                            "retryable": False,
                            "requires_state_check": True,
                        },
                        error=True,
                    )
                    _LOGGER.warning(
                        "Tool result serialization failed (%s)", type(err).__name__
                    )

        return _make_tool_result_content(
            agent_id=self.entity_id,
            tool_call_id=tool_input.id,
            tool_name=tool_input.tool_name,
            data=outcome.data,
            error=outcome.error,
        )

    async def _async_run_background_tool(
        self,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> None:
        """Observe delayed failures without mutating a completed conversation."""
        try:
            result = await get_function(function_config["type"]).execute(
                self.hass, function_config, arguments, llm_context, exposed_entities
            )
            failed = result_is_error(result, function_config)
            _LOGGER.log(
                logging.WARNING if failed else logging.DEBUG,
                "Background tool finished (status=%s)",
                "failed_or_unknown" if failed else "completed",
            )
        except asyncio.CancelledError:
            _LOGGER.debug("Background tool cancelled (outcome=unknown)")
            raise
        except Exception as err:
            if is_fatal_error(err) or self.hass.is_stopping:
                raise
            _LOGGER.warning(
                "Background tool failed (%s, outcome=unknown)", type(err).__name__
            )

    def should_run_in_background(self, arguments: dict[str, Any]) -> bool:
        """Check if function needs delay."""
        return isinstance(arguments, dict) and arguments.get("delay") is not None

    def get_delayed_function_config(
        self, function_config: dict[str, Any], arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute function with delay."""
        # create a composite function with delay in script function
        return {
            "type": "composite",
            "sequence": [
                {
                    "type": "script",
                    "sequence": [{"delay": arguments["delay"]}],
                },
                function_config,
            ],
        }

    async def _truncate_message_history(self, chat_log: conversation.ChatLog) -> None:
        """Truncate message history based on strategy."""
        options = self.subentry.data
        strategy = options.get(
            CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY
        )

        if strategy == "clear":
            # Keep only system prompt and last user message
            # This is handled by refreshing the LLM data
            _LOGGER.info("Context threshold exceeded, conversation history cleared")
            last_user_message_index = None
            messages = chat_log.content
            for i in reversed(range(len(messages))):
                if isinstance(messages[i], conversation.UserContent):
                    last_user_message_index = i
                    break

            if last_user_message_index is not None:
                del messages[1:last_user_message_index]
