"""路由与编排集成测试 — 适配重构后结构"""
import pytest
from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage, AIMessage

from app.agent.nodes import (
    route_after_supervisor,
    _extract_json,
    create_agent_node,
)
from app.agent.registry import AgentRegistry, bootstrap_agents
from app.agent.workers.chat_agent import ChatAgent
from app.core import deps


@pytest.fixture(autouse=True)
def reset_state():
    deps.reset_all_deps()
    AgentRegistry.clear()
    yield
    deps.reset_all_deps()
    AgentRegistry.clear()


class TestJSONExtraction:
    def test_plain_json(self):
        assert _extract_json('{"next_agent": "chat_agent"}')["next_agent"] == "chat_agent"

    def test_code_block(self):
        assert _extract_json('```json\n{"next_agent": null}\n```')["next_agent"] is None

    def test_embedded(self):
        assert _extract_json('结果：{"next_agent": "rag_agent"}')["next_agent"] == "rag_agent"

    def test_no_json(self):
        assert _extract_json("纯文本") == {}


class TestRouting:
    def test_route_to_registered_agent(self):
        bootstrap_agents()
        state = {"messages": [], "next_agent": "chat_agent"}
        assert route_after_supervisor(state) == "chat_agent"

    def test_route_null_to_finalize(self):
        bootstrap_agents()
        state = {"messages": [], "next_agent": None}
        assert route_after_supervisor(state) == "finalize"

    def test_route_unknown_to_finalize(self):
        bootstrap_agents()
        state = {"messages": [], "next_agent": "unknown_xyz"}
        assert route_after_supervisor(state) == "finalize"


class TestFinalizeNode:
    def test_passthrough_ai_message(self):
        import asyncio
        from app.agent.nodes import finalize_node

        async def _run():
            msg = AIMessage(content="final answer")
            state = {"messages": [HumanMessage(content="q"), msg]}
            return await finalize_node(state)

        result = asyncio.run(_run())
        assert result == {}  # 已透传，不新增

    def test_fallback_when_no_ai_message(self):
        import asyncio
        from app.agent.nodes import finalize_node

        async def _run():
            state = {
                "messages": [HumanMessage(content="q")],
                "agent_outputs": {"chat_agent": {"content": "generated answer"}},
            }
            return await finalize_node(state)

        result = asyncio.run(_run())
        assert "messages" in result
        assert "generated answer" in result["messages"][0].content


class TestAgentNodeFactory:
    def test_node_sets_current_agent(self):
        import asyncio

        async def _run():
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = AIMessage(content="ok")

            agent = ChatAgent(model=mock_llm)
            node_fn = create_agent_node(agent)

            state = {"session_id": "t", "messages": [HumanMessage(content="hi")]}
            return await node_fn(state)

        result = asyncio.run(_run())
        assert result["current_agent"] == "chat_agent"
