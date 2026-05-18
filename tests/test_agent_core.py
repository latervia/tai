"""Agent 核心组件单元测试 — 适配重构后的结构"""
import uuid
from unittest.mock import MagicMock, AsyncMock

import pytest

import app.deps as deps
from app.domain.agent.human.approval import ApprovalManager
from app.domain.agent.tools.registry import ToolRegistry
from app.domain.agent.types import (
    RiskLevel, ToolDef, )
from app.domain.agent.workers.chat_agent import ChatAgent
from app.domain.agent.workers.rag_agent import RAGAgent
from app.domain.agent.workers.registry import AgentRegistry, bootstrap_agents


# ── 每个测试前重置全局状态 ─────────────────────────────

@pytest.fixture(autouse=True)
def reset_deps():
    deps.reset_all_deps()
    ToolRegistry.clear()
    AgentRegistry.clear()
    yield
    deps.reset_all_deps()
    ToolRegistry.clear()
    AgentRegistry.clear()


class TestToolDef:
    def test_risk_level_default(self):
        td = ToolDef(name="t", description="d", fn=lambda: None)
        assert td.risk_level == RiskLevel.LOW

    def test_high_risk_tool(self):
        td = ToolDef(name="t", description="d", fn=lambda: None, risk_level=RiskLevel.HIGH)
        assert td.risk_level == RiskLevel.HIGH


class TestToolRegistry:
    def test_register_and_get(self):
        ToolRegistry.register("echo", lambda x: x, description="回显", permissions=["read"])
        tool = ToolRegistry.get("echo")
        assert tool is not None
        assert tool.fn("hello") == "hello"

    def test_risk_level_passed_through(self):
        ToolRegistry.register("danger", lambda: None, risk_level=RiskLevel.CRITICAL)
        tool = ToolRegistry.get("danger")
        assert tool.risk_level == RiskLevel.CRITICAL

    def test_get_for_agent_with_permissions(self):
        ToolRegistry.register("read_file", lambda: "data", permissions=["read_file"])
        ToolRegistry.register("write_file", lambda: None, permissions=["write_file"])
        ToolRegistry.grant("reader", ["read_file"])
        assert len(ToolRegistry.get_for_agent("reader")) == 1

    def test_no_permissions_returns_empty(self):
        ToolRegistry.register("admin_tool", lambda: None, permissions=["admin"])
        assert ToolRegistry.get_for_agent("guest") == []


class TestApprovalManager:
    def setup_method(self):
        self.am = ApprovalManager()

    def test_full_lifecycle(self):
        rid = str(uuid.uuid4())
        self.am.request(rid, "delete", {"id": 1}, "test", RiskLevel.HIGH)
        assert len(self.am.list_pending()) == 1

        self.am.approve(rid)
        assert self.am.get(rid).status == "approved"

    def test_reject(self):
        rid = str(uuid.uuid4())
        self.am.request(rid, "write", {}, "test", RiskLevel.MEDIUM)
        self.am.reject(rid, "不需要")
        assert self.am.get(rid).status == "rejected"

    def test_cleanup(self):
        rid = str(uuid.uuid4())
        self.am.request(rid, "op", {}, "", RiskLevel.LOW)
        self.am.cleanup(rid)
        assert self.am.get(rid) is None


class TestAgentRegistry:
    def test_bootstrap_registers_expected(self):
        bootstrap_agents()
        agents = AgentRegistry.list_agents()
        names = [a["name"] for a in agents]
        assert "chat_agent" in names
        assert "rag_agent" in names

    def test_bootstrap_is_idempotent(self):
        bootstrap_agents()
        count1 = len(AgentRegistry.list_agents())
        bootstrap_agents()
        count2 = len(AgentRegistry.list_agents())
        assert count1 == count2

    def test_get_cls(self):
        bootstrap_agents()
        cls = AgentRegistry.get_cls("chat_agent")
        assert cls is ChatAgent


class TestChatAgent:
    def test_metadata(self):
        assert ChatAgent.AGENT_NAME == "chat_agent"
        assert len(ChatAgent.AGENT_DESCRIPTION) > 0

    def test_invoke_with_mock_llm(self):
        import asyncio
        from langchain_core.messages import AIMessage, HumanMessage

        async def _run():
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = AIMessage(content="你好，有什么可以帮你的？")

            agent = ChatAgent(model=mock_llm)
            state = {"session_id": "test", "messages": [HumanMessage(content="你好")]}
            return await agent(state)

        result = asyncio.run(_run())
        assert "messages" in result
        assert "有什么可以帮你的" in result["messages"][0].content


class TestRAGAgent:
    def test_metadata(self):
        assert RAGAgent.AGENT_NAME == "rag_agent"

    def test_tools_with_search(self):
        mock_llm = MagicMock()
        search = MagicMock()
        agent = RAGAgent(model=mock_llm, search_tool=search)
        assert len(agent.tools) == 1


class TestDeps:
    def test_singletons(self):
        tc1 = deps.get_trace_collector()
        tc2 = deps.get_trace_collector()
        assert tc1 is tc2

    def test_reset(self):
        tc1 = deps.get_trace_collector()
        deps.reset_all_deps()
        tc2 = deps.get_trace_collector()
        assert tc1 is not tc2  # reset 后是新实例
