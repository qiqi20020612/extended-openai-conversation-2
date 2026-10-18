"""I/O adapter failures, configuration reuse, privacy and local process cleanup."""

import asyncio
from contextlib import asynccontextmanager
import errno
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiohttp import ClientResponseError
import pytest

from custom_components.extended_openai_conversation.functions import get_function
from tests.test_tool_argument_retry import _answer, _calls, _results, _tool
from tests.test_tool_execution import actual_entity, assert_paired, tool_config


def http_transport(monkeypatch, replies):
    requests = []

    class Session:
        def request(self, method, url, **kwargs):
            requests.append({"method": method, "url": url, **kwargs})
            status, body = replies[len(requests) - 1]

            @asynccontextmanager
            async def response():
                if kwargs.get("raise_for_status") and status >= 400:
                    raise ClientResponseError(
                        SimpleNamespace(real_url=url),
                        (),
                        status=status,
                        message="private-password-response",
                        headers={},
                    )
                yield SimpleNamespace(
                    status=status,
                    charset="utf-8",
                    headers={"Content-Type": "text/html"},
                    text=AsyncMock(return_value=body),
                )

            return response()

    monkeypatch.setattr(
        "custom_components.extended_openai_conversation.functions.web.async_get_clientsession",
        lambda *args, **kwargs: Session(),
    )
    return requests


@pytest.mark.parametrize("kind", ["write_file", "edit_file", "bash", "rest", "scrape"])
async def test_input_template_error_is_not_executed_and_can_be_corrected(
    hass, monkeypatch, tmp_path, kind
):
    target = tmp_path / "extended_openai_conversation" / "effect.txt"
    target.write_text("before")
    if kind in {"rest", "scrape"}:
        config = {
            "method": "POST",
            "resource_template": "https://example.test/{{ target.path }}",
        }
        if kind == "scrape":
            config["sensor"] = [{"name": "Title", "select": "h1"}]
        requests = http_transport(monkeypatch, [(200, "<h1>after</h1>")])
    elif kind == "bash":
        config = {"command": "printf '{{ target.path }}'"}
    else:
        config = {"path": "{{ target.path }}"}
        config.update(
            {"content": "after"}
            if kind == "write_file"
            else {"old_text": "before", "new_text": "after"}
        )
    configured = tool_config(kind=kind, **config)
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool("{}", call_id="invalid")),
            _calls(_tool('{"target":{"path":"effect.txt"}}', call_id="fixed")),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, [configured], [])

    result = _results(log)[0].result
    assert result.error and result.data["execution_status"] == "not_executed"
    assert result.data["retryable"]
    assert not _results(log)[1].result.error
    if kind in {"rest", "scrape"}:
        assert len(requests) == 1
    if kind in {"write_file", "edit_file"}:
        assert target.read_text() == "after"
    assert_paired(log, create)


async def test_shell_os_permission_error_does_not_echo_filename(
    hass, monkeypatch, caplog
):
    configured = tool_config(kind="bash", command="printf ok")
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError(errno.EACCES, "private-token", "private-filename")
        ),
    )

    result = await get_function("bash").execute(
        hass, configured["function"], {}, None, []
    )

    assert result["permission_denied"] and result["execution_status"] == "not_executed"
    assert "private-" not in str(result) + caplog.text


@pytest.mark.parametrize("kind", ["rest", "scrape"])
async def test_result_template_failure_is_a_tool_error_and_post_cannot_replay(
    hass, monkeypatch, caplog, kind
):
    requests = http_transport(monkeypatch, [(200, "<h1>private-token</h1>")])
    config = {
        "resource": "https://example.test",
        "method": "POST",
        "value_template": "{{ absent.value }}",
    }
    if kind == "scrape":
        config["sensor"] = [{"name": "Title", "select": "h1"}]
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool("{}", call_id="one")),
            _calls(_tool("{}", call_id="two")),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, [tool_config(kind=kind, **config)], [])

    assert len(requests) == 1
    first = _results(log)[0].result
    assert first.error and first.data["execution_status"] == "unknown"
    assert _results(log)[1].result == first
    assert "private-token" not in str(first) + caplog.text
    assert_paired(log, create)


async def test_rest_templates_are_not_mutated_between_normal_calls(hass, monkeypatch):
    requests = http_transport(monkeypatch, [(200, "first"), (200, "second")])
    configured = tool_config(
        kind="rest",
        method="POST",
        resource_template="https://example.test/{{ name }}",
        payload_template="{{ name }}",
    )
    before = dict(configured["function"])
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool('{"name":"one"}', call_id="one")),
            _calls(_tool('{"name":"two"}', call_id="two")),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, [configured], [])

    assert [r["url"] for r in requests] == [
        "https://example.test/one",
        "https://example.test/two",
    ]
    assert [r["data"] for r in requests] == ["one", "two"]
    assert all(r["raise_for_status"] for r in requests)
    assert configured["function"] == before
    assert all(not r.result.error for r in _results(log))
    assert_paired(log, create)


@pytest.mark.parametrize(
    "method,status", [("GET", 401), ("GET", 403), ("GET", 500), ("POST", 500)]
)
async def test_http_error_is_safe_and_post_is_not_replayed(
    hass, monkeypatch, caplog, method, status
):
    requests = http_transport(monkeypatch, [(status, "private-password-response")])
    configured = tool_config(
        kind="rest", method=method, resource_template="https://example.test/{{ name }}"
    )
    streams = [_calls(_tool('{"name":"one"}', call_id="one"))]
    if method == "POST":
        streams.append(_calls(_tool('{"name":"two"}', call_id="rewritten")))
    streams.append(_answer())
    entity, log, create = actual_entity(hass, streams)

    await entity._async_handle_chat_log(log, [configured], [])

    assert len(requests) == 1
    result = _results(log)[0].result
    assert result.error and result.data["http_status"] == status
    assert "private-password-response" not in str(log.as_dict()) + caplog.text
    if status in {401, 403}:
        assert result.data["permission_denied"] and not result.data["retryable"]
    if method == "POST":
        assert result.data["execution_status"] == "unknown"
        assert _results(log)[1].result.data["error"] == "UnsafeRetryBlocked"
    assert_paired(log, create)


async def test_scrape_fetches_once_and_http_failure_is_not_empty_success(
    hass, monkeypatch
):
    requests = http_transport(
        monkeypatch,
        [(500, "private-password-response"), (200, "<h1>Recovered title</h1>")],
    )
    configured = tool_config(
        kind="scrape",
        resource_template="https://example.test/{{ name }}",
        sensor=[{"name": "Title", "select": "h1"}],
    )
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool('{"name":"one"}', call_id="one")),
            _calls(_tool('{"name":"two"}', call_id="two")),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, [configured], [])

    assert len(requests) == 2
    assert _results(log)[0].result.error
    assert _results(log)[1].result.data == {"result": "Recovered title"}
    assert "resource_template" in configured["function"]
    assert_paired(log, create)


@pytest.mark.parametrize("kind", ["read_file", "write_file", "edit_file"])
@pytest.mark.parametrize("escape", ["prefix", "symlink", "config_override"])
async def test_file_policy_cannot_escape_via_rewritten_paths(
    hass, tmp_path, kind, escape, caplog
):
    outside = tmp_path / "extended_openai_conversation_private-token"
    outside.mkdir()
    target = outside / "private-password.txt"
    target.write_text("private-content")
    if escape == "symlink":
        link = tmp_path / "extended_openai_conversation" / "link"
        link.symlink_to(outside, target_is_directory=True)
        filename = "link/private-password.txt"
    else:
        filename = str(target)
    config = {"path": "{{ filename }}", "allow_dir": ["{{ config_dir }}/safe"]}
    if kind == "write_file":
        config["content"] = "replacement"
    if kind == "edit_file":
        config.update(old_text="private-content", new_text="replacement")
    configured = tool_config(kind=kind, **config)
    arguments = {"filename": filename, "config_dir": str(outside.parent)}
    if escape == "config_override":
        # The model must not replace the trusted config_dir used by allow_dir.
        (outside / "safe").mkdir()
        target = outside / "safe" / "private-password.txt"
        target.write_text("private-content")
        arguments.update(filename=str(target), config_dir=str(outside))

    result = await get_function(kind).execute(
        hass, configured["function"], arguments, None, []
    )

    assert result["permission_denied"] and not result["retryable"]
    assert "private-password" not in str(result) + caplog.text
    assert target.read_text() == "private-content"
    hass.async_add_executor_job.assert_not_awaited()


async def test_partial_file_write_error_is_unknown_and_blocks_rewrite(
    hass, tmp_path, monkeypatch, caplog
):
    configured = tool_config(
        kind="write_file", path="effect.txt", content="{{ content }}"
    )
    writes = []

    async def executor(func, *args):
        if getattr(getattr(func, "func", None), "__name__", "") == "write_text":
            writes.append(True)
            (tmp_path / "extended_openai_conversation" / "effect.txt").write_text(
                "partial"
            )
            raise PermissionError(errno.EACCES, "private-password")
        return func(*args)

    hass.async_add_executor_job.side_effect = executor
    entity, log, create = actual_entity(
        hass,
        [
            _calls(_tool('{"content":"first"}', call_id="first")),
            _calls(_tool('{"content":"corrected"}', call_id="second")),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, [configured], [])

    assert writes == [True]
    assert _results(log)[0].result.data["execution_status"] == "unknown"
    assert _results(log)[1].result.data["error"] == "UnsafeRetryBlocked"
    assert (
        tmp_path / "extended_openai_conversation" / "effect.txt"
    ).read_text() == "partial"
    assert "private-password" not in str(log.as_dict()) + caplog.text
    assert_paired(log, create)


async def test_sqlite_missing_table_is_feedback_and_empty_single_result_is_valid(
    hass, temp_db_path, caplog
):
    configured = tool_config(
        kind="sqlite", db_url=str(temp_db_path), query="{{query}}", single=True
    )
    entity, log, create = actual_entity(
        hass,
        [
            _calls(
                _tool(
                    json.dumps({"query": "SELECT private_password FROM absent"}),
                    call_id="bad",
                )
            ),
            _calls(
                _tool(
                    json.dumps({"query": "SELECT * FROM states WHERE 0"}),
                    call_id="empty",
                )
            ),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, [configured], [])

    assert _results(log)[0].result.error
    assert _results(log)[0].result.data["error"] == "OperationalError"
    assert _results(log)[1].result.data == {"result": "{}"}
    assert "SELECT private_password" not in caplog.text
    assert_paired(log, create)


async def test_sqlite_trusted_exposure_helpers_cannot_be_overridden(hass, temp_db_path):
    configured = tool_config(
        kind="sqlite",
        db_url=str(temp_db_path),
        query="{{ is_exposed('light.private') }}",
    )
    entity, log, create = actual_entity(
        hass,
        [
            _calls(
                _tool(
                    '{"is_exposed":true,"exposed_entities":[{"entity_id":"light.private"}]}'
                )
            ),
            _answer(),
        ],
    )

    await entity._async_handle_chat_log(log, [configured], [])

    # False is rendered by the trusted helper; the supplied boolean cannot be called.
    assert _results(log)[0].result.data["error"] == "OperationalError"
    assert_paired(log, create)


@pytest.mark.parametrize("cancel", [False, True])
async def test_shell_timeout_and_cancellation_reap_process_group(
    hass, tmp_path, monkeypatch, cancel
):
    configured = tool_config(
        kind="bash",
        command="printf before > effect.txt; sleep 5; printf after >> effect.txt",
    )
    processes = []
    original = asyncio.create_subprocess_shell

    async def create(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_shell", create)
    task = asyncio.create_task(
        get_function("bash").execute(
            hass,
            configured["function"],
            {"timeout": 0.03 if not cancel else 5},
            None,
            [],
        )
    )
    effect = tmp_path / "extended_openai_conversation" / "effect.txt"
    if cancel:
        for _ in range(200):
            if effect.exists():
                break
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        assert result["execution_status"] == "unknown" and not result["retryable"]
    assert len(processes) == 1 and processes[0].returncode is not None
    assert effect.read_text() == "before"


async def test_nonzero_shell_exit_preserves_output_and_is_not_safe_to_repeat(hass):
    configured = tool_config(kind="bash", command="printf before; exit 2")

    result = await get_function("bash").execute(
        hass, configured["function"], {}, None, []
    )

    assert result["exit_code"] == 2 and result["stdout"] == "before"
    assert result["execution_status"] == "unknown" and not result["retryable"]
