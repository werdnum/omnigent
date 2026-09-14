"""Pi auto-retry events belong to the same Omnigent turn."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from omnigent.inner.executor import (
    ExecutorError,
    ReasoningChunk,
    TextChunk,
    ToolCallComplete,
    TurnComplete,
)
from omnigent.inner.pi_executor import PiExecutor, _PiRpcSession, _PiSessionState


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
    monkeypatch.setattr(rpc, "close", AsyncMock(wraps=rpc.close))
    executor._session_states["__default__"] = _PiSessionState(rpc=rpc)
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


@pytest.mark.parametrize("message_end_present", [True, False])
@pytest.mark.parametrize("terminal_message_present", [True, False])
async def test_exhausted_retry_reports_final_error(
    monkeypatch: pytest.MonkeyPatch,
    message_end_present: bool,
    terminal_message_present: bool,
) -> None:
    final = _error("503 retry exhausted")
    events = [
        *_retry_events("503 first attempt"),
        {"type": "agent_start"},
        *([{"type": "message_end", "message": final}] if message_end_present else []),
        {
            "type": "agent_end",
            "messages": [final] if terminal_message_present else [],
            "willRetry": False,
        },
        {"type": "auto_retry_end", "success": False, "finalError": final["errorMessage"]},
    ]
    result, _ = await _run(monkeypatch, events)
    assert [event.message for event in result if isinstance(event, ExecutorError)] == [
        "503 retry exhausted"
        if message_end_present or terminal_message_present
        else "503 first attempt"
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
        *_retry_events("503 retry interrupted"),
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "Partial"},
        },
    ]
    result, _ = await _run(monkeypatch, events)
    assert [event.message for event in result if isinstance(event, ExecutorError)] == [
        "503 retry interrupted"
    ]
    assert not any(isinstance(event, TurnComplete) for event in result)


@pytest.mark.parametrize("kind", ["text_delta", "thinking_delta"])
async def test_partial_retry_stops_process_without_mixing_answers(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    def text(delta: str) -> dict:
        return {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": delta},
        }

    events = [
        text("Before tool. "),
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "toolUse"}},
        {"type": "tool_execution_end", "toolName": "diagnostic", "result": "ok"},
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": kind, "delta": "Partial"},
        },
        *_retry_events("503 first attempt"),
        {"type": "agent_start"},
        text("Recovered"),
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop"}},
        {"type": "auto_retry_end", "success": True, "attempt": 1},
        {"type": "agent_end", "messages": [], "willRetry": False},
    ]
    result, rpc = await _run(monkeypatch, events)
    expected_text = ["Before tool. ", "Partial"] if kind == "text_delta" else ["Before tool. "]
    assert [event.text for event in result if isinstance(event, TextChunk)] == expected_text
    assert [event.delta for event in result if isinstance(event, ReasoningChunk)] == (
        ["Partial"] if kind == "thinking_delta" else []
    )
    assert [event.message for event in result if isinstance(event, ExecutorError)] == [
        "503 first attempt"
    ]
    assert not any(isinstance(event, TurnComplete) for event in result)
    assert isinstance(rpc.close, AsyncMock)
    rpc.close.assert_awaited_once()


@pytest.mark.parametrize(
    ("stop", "agent_end", "retrying"),
    [
        ("error", True, False),
        ("error", False, False),
        ("aborted", False, False),
        ("error", True, True),
        ("error", False, True),
    ],
)
async def test_terminal_failure_preserves_partial_output(
    monkeypatch: pytest.MonkeyPatch, stop: str, agent_end: bool, retrying: bool
) -> None:
    events = _retry_events("503 first attempt") if retrying else []
    events.extend(
        [
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": "Partial"},
            },
            {
                "type": "message_end",
                "message": {"role": "assistant", "stopReason": stop, "errorMessage": "terminal"},
            },
        ]
    )
    if agent_end:
        events.append({"type": "agent_end", "messages": [], "willRetry": False})
    result, _ = await _run(monkeypatch, events)
    assert [event.text for event in result if isinstance(event, TextChunk)] == ["Partial"]
    assert [event.message for event in result if isinstance(event, ExecutorError)] == ["terminal"]
    assert not any(isinstance(event, TurnComplete) for event in result)


async def test_successful_message_does_not_block_later_silent_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, rpc = await _run(
        monkeypatch,
        [
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": "Before tool. "},
            },
            {"type": "message_end", "message": {"role": "assistant", "stopReason": "toolUse"}},
            {"type": "tool_execution_end", "toolName": "diagnostic", "result": "ok"},
            *_retry_events("503 first attempt"),
            {"type": "agent_start"},
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": "Recovered"},
            },
            {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop"}},
            {"type": "auto_retry_end", "success": True, "attempt": 1},
            {"type": "agent_end", "messages": [], "willRetry": False},
        ],
    )
    assert not any(isinstance(event, ExecutorError) for event in result)
    assert [event.response for event in result if isinstance(event, TurnComplete)] == [
        "Before tool. Recovered"
    ]
    assert isinstance(rpc.close, AsyncMock)
    rpc.close.assert_not_awaited()


@pytest.mark.parametrize("kind", ["text_delta", "thinking_delta"])
async def test_output_streams_before_message_end(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setattr("omnigent.inner.pi_executor._find_pi_cli", lambda: "/unused/pi")
    executor = PiExecutor()
    rpc = _PiRpcSession()
    rpc._line_queue = asyncio.Queue()
    rpc._line_queue.put_nowait(
        json.dumps(
            {"type": "message_update", "assistantMessageEvent": {"type": kind, "delta": "Now"}}
        )
    )
    monkeypatch.setattr(rpc, "send_command", AsyncMock())
    monkeypatch.setattr(executor, "_ensure_rpc", AsyncMock(return_value=rpc))
    stream = executor.run_turn([{"role": "user", "content": "diagnostic"}], [], "")
    try:
        event = await asyncio.wait_for(anext(stream), timeout=1)
        if kind == "text_delta":
            assert isinstance(event, TextChunk)
            assert event.text == "Now"
        else:
            assert isinstance(event, ReasoningChunk)
            assert event.delta == "Now"
    finally:
        await stream.aclose()
        await rpc.close()
