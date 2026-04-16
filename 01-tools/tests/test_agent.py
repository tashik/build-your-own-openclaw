"""Regression tests for AgentSession.chat() tool_calls assignment.

Tests that 'tool_calls' is only present in the assistant message when the LLM
actually returns tool calls, and that the value is correctly assigned when present.
This guards against the bug fixed in commit 99c1873 where 'tool_calls' was always
included in the assistant message dict (even as an empty list), which could cause
issues with LLM providers that reject empty tool_calls.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass

from mybot.core.agent import Agent, AgentSession
from mybot.core.session_state import SessionState
from mybot.provider.llm import LLMToolCall
from mybot.tools.registry import ToolRegistry


@dataclass
class FakeAgentDef:
    """Minimal stand-in for AgentDef used by SessionState.build_messages."""
    agent_md: str = "You are a test agent."
    llm: MagicMock = None

    def __post_init__(self):
        if self.llm is None:
            self.llm = MagicMock()


def _make_session(llm_mock: AsyncMock) -> AgentSession:
    """Create an AgentSession wired to a mocked LLM provider."""
    agent = MagicMock(spec=Agent)
    agent.agent_def = FakeAgentDef()
    agent.llm = llm_mock

    state = SessionState(
        session_id="test-session",
        agent=agent,
        messages=[],
    )

    tools = ToolRegistry()
    return AgentSession(agent=agent, state=state, tools=tools)


@pytest.mark.asyncio
async def test_tool_calls_present_when_llm_returns_tool_calls():
    """When the LLM returns tool calls, 'tool_calls' must be in the assistant message."""
    tool_call = LLMToolCall(
        id="call_123",
        name="read_file",
        arguments=json.dumps({"path": "/tmp/test.txt"}),
    )

    llm_mock = AsyncMock()
    # First call: LLM returns a tool call
    llm_mock.chat = AsyncMock(
        side_effect=[
            ("Using tool...", [tool_call]),
            ("Here is the result.", []),
        ]
    )

    session = _make_session(llm_mock)
    # Mock the tool execution so _handle_tool_calls completes
    session.tools.execute_tool = AsyncMock(return_value="file contents")

    await session.chat("read the file")

    # Find the first assistant message (the one with tool calls)
    assistant_msgs = [m for m in session.state.messages if m["role"] == "assistant"]
    assert len(assistant_msgs) == 2

    first_assistant = assistant_msgs[0]
    assert "tool_calls" in first_assistant, (
        "assistant message must contain 'tool_calls' when the LLM returns tool calls"
    )
    assert len(first_assistant["tool_calls"]) == 1

    tc = first_assistant["tool_calls"][0]
    assert tc["id"] == "call_123"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "read_file"
    assert tc["function"]["arguments"] == json.dumps({"path": "/tmp/test.txt"})


@pytest.mark.asyncio
async def test_tool_calls_absent_when_llm_returns_no_tool_calls():
    """When the LLM returns no tool calls, 'tool_calls' must NOT be in the assistant message."""
    llm_mock = AsyncMock()
    llm_mock.chat = AsyncMock(return_value=("Hello! How can I help?", []))

    session = _make_session(llm_mock)

    await session.chat("hi")

    assistant_msgs = [m for m in session.state.messages if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1

    assistant_msg = assistant_msgs[0]
    assert "tool_calls" not in assistant_msg, (
        "assistant message must NOT contain 'tool_calls' when the LLM returns no tool calls"
    )


@pytest.mark.asyncio
async def test_tool_calls_correct_with_multiple_tool_calls():
    """When the LLM returns multiple tool calls, all are correctly assigned."""
    tool_calls = [
        LLMToolCall(id="call_1", name="read_file", arguments=json.dumps({"path": "a.txt"})),
        LLMToolCall(id="call_2", name="bash", arguments=json.dumps({"command": "ls"})),
    ]

    llm_mock = AsyncMock()
    llm_mock.chat = AsyncMock(
        side_effect=[
            ("Running tools...", tool_calls),
            ("Done.", []),
        ]
    )

    session = _make_session(llm_mock)
    session.tools.execute_tool = AsyncMock(return_value="ok")

    await session.chat("do stuff")

    assistant_msgs = [m for m in session.state.messages if m["role"] == "assistant"]
    first_assistant = assistant_msgs[0]

    assert "tool_calls" in first_assistant
    assert len(first_assistant["tool_calls"]) == 2
    assert first_assistant["tool_calls"][0]["id"] == "call_1"
    assert first_assistant["tool_calls"][0]["function"]["name"] == "read_file"
    assert first_assistant["tool_calls"][1]["id"] == "call_2"
    assert first_assistant["tool_calls"][1]["function"]["name"] == "bash"


@pytest.mark.asyncio
async def test_assistant_message_content_preserved_regardless_of_tool_calls():
    """Content field is always set correctly, with or without tool calls."""
    tool_call = LLMToolCall(
        id="call_x", name="bash", arguments=json.dumps({"command": "echo hi"}),
    )

    llm_mock = AsyncMock()
    llm_mock.chat = AsyncMock(
        side_effect=[
            ("Thinking...", [tool_call]),
            ("Final answer.", []),
        ]
    )

    session = _make_session(llm_mock)
    session.tools.execute_tool = AsyncMock(return_value="hi")

    result = await session.chat("say hi")

    assistant_msgs = [m for m in session.state.messages if m["role"] == "assistant"]

    # First assistant message (with tool call)
    assert assistant_msgs[0]["content"] == "Thinking..."
    assert "tool_calls" in assistant_msgs[0]

    # Second assistant message (no tool call)
    assert assistant_msgs[1]["content"] == "Final answer."
    assert "tool_calls" not in assistant_msgs[1]

    assert result == "Final answer."
