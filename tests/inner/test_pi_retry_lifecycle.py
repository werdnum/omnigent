"""Pi auto-retry events belong to the same Omnigent turn."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from omnigent.inner.executor import ExecutorError, TextChunk, ToolCallComplete, TurnComplete
from omnigent.inner.pi_executor import PiExecutor, _PiRpcSession


def _error(message: str) -> dict:
    return {"role": "assistant", "content": [], "stopReason": "error", "errorMessage": message}


def _retry_events(message: str) -> list[dict]:
    assistant = _error(message)
    return [
        {"type": "message_end", "message": assistant},
        {"type": "agent_end", "messages": [assistant], "willRetry": True},
        {"type": "auto_retry_start", "attempt": 1, "errorMessage": message},
    ]


async def _run(monkeypatch: pytest.MonkeyPatch, events: list[dict]) -> tuple[list, _PiRpcSession]:
    monkeypatch.setattr("omnigent.inner.pi_executor._find_pi_cli", lambda: "/unused/pi")
    executor = PiExecutor()
    rpc = _PiRpcSession()
    rpc._line_queue = asyncio.Queue()
    for event in events:
        rpc._line_queue.put_nowait(json.dumps(event))
    rpc._line_queue.put_nowait(None)
    monkeypatch.setattr(rpc, "send_command", AsyncMock())
    monkeypatch.setattr(executor, "_ensure_rpc", AsyncMock(return_value=rpc))
    result = [
        event
        async for event in executor.run_turn([{"role": "user", "content": "diagnostic"}], [], "")
    ]
    return result, rpc


@pytest.mark.parametrize("message_end_present", [True, False])
async def test_retry_continues_through_tools_to_final_answer(
    monkeypatch: pytest.MonkeyPatch, message_end_present: bool
) -> None:
    initial = _retry_events("Upstream idle timeout exceeded")
    if not message_end_present:
        initial = initial[1:]
    events = [
        *initial,
        {"type": "agent_start"},
        {"type": "tool_execution_end", "toolName": "diagnostic", "result": "ok"},
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "Recovered"},
        },
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop"}},
        {"type": "auto_retry_end", "success": True, "attempt": 1},
        {"type": "agent_end", "messages": [], "willRetry": False},
    ]
    result, _ = await _run(monkeypatch, events)
    assert not any(isinstance(event, ExecutorError) for event in result)
    assert any(isinstance(event, ToolCallComplete) for event in result)
    assert [event.text for event in result if isinstance(event, TextChunk)] == ["Recovered"]
    assert [event.response for event in result if isinstance(event, TurnComplete)] == ["Recovered"]


async def test_exhausted_retry_reports_final_error(monkeypatch: pytest.MonkeyPatch) -> None:
    final = _error("503 retry exhausted")
    events = [
        *_retry_events("503 first attempt"),
        {"type": "agent_start"},
        {"type": "message_end", "message": final},
        {"type": "agent_end", "messages": [final], "willRetry": False},
        {"type": "auto_retry_end", "success": False, "finalError": final["errorMessage"]},
    ]
    result, _ = await _run(monkeypatch, events)
    assert [event.message for event in result if isinstance(event, ExecutorError)] == [
        "503 retry exhausted"
    ]
    assert not any(isinstance(event, TurnComplete) for event in result)


async def test_cancelled_retry_ends_without_another_agent_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        *_retry_events("503 first attempt"),
        {"type": "auto_retry_end", "success": False, "finalError": "Retry cancelled"},
    ]
    result, _ = await _run(monkeypatch, events)
    assert [event.message for event in result if isinstance(event, ExecutorError)] == [
        "Retry cancelled"
    ]
    assert not any(isinstance(event, TurnComplete) for event in result)


async def test_stale_retry_end_does_not_fail_next_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "auto_retry_end", "success": False, "finalError": "previous turn"},
        {"type": "agent_start"},
        {"type": "agent_end", "messages": [{"role": "assistant", "content": "New answer"}]},
    ]
    result, _ = await _run(monkeypatch, events)
    assert not any(isinstance(event, ExecutorError) for event in result)
    assert [event.response for event in result if isinstance(event, TurnComplete)] == [
        "New answer"
    ]


async def test_eof_during_retry_is_not_partial_success(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "Partial"},
        },
        *_retry_events("503 retry interrupted"),
    ]
    result, _ = await _run(monkeypatch, events)
    assert [event.message for event in result if isinstance(event, ExecutorError)] == [
        "503 retry interrupted"
    ]
    assert not any(isinstance(event, TurnComplete) for event in result)


async def test_retry_discards_only_failed_message_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def text(delta: str) -> dict:
        return {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": delta},
        }

    events = [
        text("Before tool. "),
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "toolUse"}},
        {"type": "tool_execution_end", "toolName": "diagnostic", "result": "ok"},
        text("Discard this partial answer"),
        *_retry_events("503 first attempt"),
        {"type": "agent_start"},
        text("Recovered"),
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop"}},
        {"type": "auto_retry_end", "success": True, "attempt": 1},
        {"type": "agent_end", "messages": [], "willRetry": False},
    ]
    result, _ = await _run(monkeypatch, events)
    assert [event.text for event in result if isinstance(event, TextChunk)] == [
        "Before tool. ",
        "Recovered",
    ]
    assert [event.response for event in result if isinstance(event, TurnComplete)] == [
        "Before tool. Recovered"
    ]
    assert not any(isinstance(event, ExecutorError) for event in result)
