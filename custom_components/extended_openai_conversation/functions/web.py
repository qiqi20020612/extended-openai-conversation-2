"""Web functions for HTTP API calls and HTML scraping."""

from __future__ import annotations

import logging
from typing import Any, cast

import aiohttp
from bs4 import BeautifulSoup
import probatio as vol

from homeassistant.components import rest, scrape
from homeassistant.const import (
    CONF_ATTRIBUTE,
    CONF_METHOD,
    CONF_NAME,
    CONF_PAYLOAD,
    CONF_RESOURCE,
    CONF_RESOURCE_TEMPLATE,
    CONF_TIMEOUT,
    CONF_VALUE_TEMPLATE,
    CONF_VERIFY_SSL,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import config_validation as cv, llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.template import Template
from homeassistant.util.ssl import SSLCipherList

from ..const import CONF_PAYLOAD_TEMPLATE
from .base import Function
from .errors import EXPECTED_TOOL_ERRORS, execution_error, is_fatal_error

_LOGGER = logging.getLogger(__name__)


class _CheckedToolSession:
    """Check HTTP status on this RestData only, without modifying HA's shared session."""

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        self.hass = hass
        self.config = config

    def request(self, *args: Any, **kwargs: Any) -> Any:
        session = async_get_clientsession(
            self.hass,
            verify_ssl=self.config[CONF_VERIFY_SSL],
            ssl_cipher=SSLCipherList(
                self.config.get(
                    rest.const.CONF_SSL_CIPHER_LIST, rest.const.DEFAULT_SSL_CIPHER_LIST
                )
            ),
        )
        return session.request(*args, **kwargs, raise_for_status=True)


async def _update_rest_data(rest_data: rest.data.RestData) -> None:
    """HA stores transport errors instead of raising; do not mistake None for success."""
    await rest_data.async_update(log_errors=False)
    if isinstance(rest_data.last_exception, Exception):
        raise rest_data.last_exception


def get_rest_data(
    hass: HomeAssistant, rest_config: dict[str, Any], arguments: dict[str, Any]
) -> rest.data.RestData:
    """Create RestData from config with template rendering."""
    rest_config = dict(rest_config)
    rest_config.setdefault(CONF_METHOD, rest.const.DEFAULT_METHOD)
    rest_config.setdefault(CONF_VERIFY_SSL, rest.const.DEFAULT_VERIFY_SSL)
    rest_config.setdefault(CONF_TIMEOUT, rest.data.DEFAULT_TIMEOUT)
    rest_config.setdefault(rest.const.CONF_ENCODING, rest.const.DEFAULT_ENCODING)

    resource_template: Template | None = rest_config.get(CONF_RESOURCE_TEMPLATE)
    if resource_template is not None:
        rest_config.pop(CONF_RESOURCE_TEMPLATE)
        rest_config[CONF_RESOURCE] = resource_template.async_render(
            arguments, parse_result=False
        )

    payload_template: Template | None = rest_config.get(CONF_PAYLOAD_TEMPLATE)
    if payload_template is not None:
        rest_config.pop(CONF_PAYLOAD_TEMPLATE)
        rest_config[CONF_PAYLOAD] = payload_template.async_render(
            arguments, parse_result=False
        )

    rest_data = rest.create_rest_data_from_config(hass, rest_config)
    # HA 2026.10 exposes no per-request status-check option on RestData. The
    # adapter preserves HA auth/encoding/SSL behavior and makes exactly one request.
    rest_data._session = cast(
        aiohttp.ClientSession, _CheckedToolSession(hass, rest_config)
    )
    return rest_data


class RestFunction(Function):
    """REST tool for HTTP API calls."""

    def __init__(self) -> None:
        """Initialize Rest tool."""
        super().__init__(
            vol.Schema(rest.RESOURCE_SCHEMA).extend(
                {
                    vol.Optional("value_template"): cv.template,
                    vol.Optional("payload_template"): cv.template,
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
        """Execute REST API call."""
        try:
            rest_data = get_rest_data(hass, function_config, arguments)
        except EXPECTED_TOOL_ERRORS as err:
            if is_fatal_error(err) or hass.is_stopping:
                raise
            return execution_error(err, status="not_executed")

        await _update_rest_data(rest_data)
        value = rest_data.data_without_xml()
        if value is None:
            raise HomeAssistantError("Remote data is unavailable")
        value_template = function_config.get(CONF_VALUE_TEMPLATE)

        if value is not None and value_template is not None:
            value = value_template.async_render_with_possible_json_value(
                value, None, arguments
            )
            if value is None:
                raise TemplateError("Result template failed")

        return value


class ScrapeFunction(Function):
    """Scrape tool for HTML content extraction."""

    def __init__(self) -> None:
        """Initialize Scrape tool."""
        super().__init__(
            scrape.COMBINED_SCHEMA.extend(
                {
                    vol.Optional("value_template"): cv.template,
                    vol.Optional("payload_template"): cv.template,
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
        """Execute web scraping."""
        try:
            rest_data = get_rest_data(hass, function_config, arguments)
        except EXPECTED_TOOL_ERRORS as err:
            if is_fatal_error(err) or hass.is_stopping:
                raise
            return execution_error(err, status="not_executed")
        # Refresh once. A coordinator can swallow the error or fetch again,
        # which is unsafe when a scrape is configured with a side-effecting method.
        await _update_rest_data(rest_data)
        if not isinstance(rest_data.data, str):
            raise HomeAssistantError("Remote data is unavailable")
        headers = rest_data.headers
        content_type = headers.get("Content-Type", "") if headers else ""
        xml = isinstance(content_type, str) and content_type.startswith(
            scrape.coordinator.XML_MIME_TYPES
        )
        raw_data = rest_data.data.lstrip()
        if raw_data.startswith("<?xml"):
            after_xml = raw_data.partition("?>")[2].lstrip()
            if after_xml.lower().startswith(("<!doctype html", "<html")):
                raw_data = after_xml
            else:
                xml = True
        soup = await hass.async_add_executor_job(
            BeautifulSoup, raw_data, "lxml-xml" if xml else "lxml"
        )

        new_arguments = dict(arguments)

        for sensor_config in function_config["sensor"]:
            name: Template = sensor_config.get(CONF_NAME)
            value = self._async_update_from_rest_data(soup, sensor_config, arguments)
            if value is None:
                raise HomeAssistantError("Scrape selection returned no value")
            new_arguments["value"] = value
            if name:
                new_arguments[name.async_render(arguments, parse_result=False)] = value

        result = new_arguments["value"]
        value_template = function_config.get(CONF_VALUE_TEMPLATE)

        if value_template is not None:
            result = value_template.async_render_with_possible_json_value(
                result, None, new_arguments
            )
            if result is None:
                raise TemplateError("Result template failed")

        return result

    def _async_update_from_rest_data(
        self,
        data: BeautifulSoup,
        sensor_config: dict[str, Any],
        arguments: dict[str, Any],
    ) -> Any:
        """Update state from the rest data."""
        value = self._extract_value(data, sensor_config)
        value_template = sensor_config.get(CONF_VALUE_TEMPLATE)

        if value_template is not None:
            value = value_template.async_render_with_possible_json_value(
                value, None, arguments
            )

        return value

    def _extract_value(self, data: BeautifulSoup, sensor_config: dict[str, Any]) -> Any:
        """Parse the html extraction in the executor."""
        value: str | list[str] | None
        select = sensor_config[scrape.const.CONF_SELECT]
        index = sensor_config.get(scrape.const.CONF_INDEX, 0)
        attr = sensor_config.get(CONF_ATTRIBUTE)
        try:
            if attr is not None:
                value = data.select(select)[index][attr]
            else:
                tag = data.select(select)[index]
                if tag.name in ("style", "script", "template"):
                    value = tag.string
                else:
                    value = tag.text
        except IndexError:
            _LOGGER.warning("Scrape selection did not match")
            value = None
        except KeyError:
            _LOGGER.warning("Scrape attribute was not found")
            value = None
        _LOGGER.debug("Parsed scrape value (type=%s)", type(value).__name__)
        return value
