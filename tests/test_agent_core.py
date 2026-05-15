"""Agent 核心组件单元测试"""
import pytest
import asyncio
import uuid
from unittest.mock import MagicMock, AsyncMock, patch

from app.agent.types import (
    AgentThought, ActionResult, ToolDef, AgentStatus, ThoughtType, TraceEvent,
)
from app.agent.tools.registry import ToolRegistry
from app.agent.workers.chat_agent import ChatAgent
from app.agent.workers.rag_agent import RAGAgent
from app.agent.registry import AgentRegistry
from app.agent.approval import ApprovalManager, ApprovalRequest, RiskLevel
from app.agent.prompts.manager import PromptManager


class TestAgentTypes:
    """共享类型定义测试"""

    def test_agent_thought_creation(self):
        thought = AgentThought(
            type=ThoughtType.RESPOND,
            content="你好",
            confidence=0.95,
        )
        assert thought.type == ThoughtType.RESPOND
        assert thought.content == "你好"
        assert thought.tool_name is None

    def test_agent_thought_tool_call(self):
        thought = AgentThought(
            type=ThoughtType.CALL_TOOL,
            content="需要搜索",
            tool_name="search",
            tool_args={"query": "Python"},
        )
        assert thought.tool_name == "search"
        assert thought.tool_args["query"] == "Python"

    def test_action_result(self):
        result = ActionResult(success=True, output="结果", duration_ms=100.5)
        assert result.success
        assert result.duration_ms == 100.5

    def test_tool_def(self):
        def dummy():
            pass
        tool = ToolDef(name="test", description="测试", fn=dummy, permissions=["read"])
        assert tool.permissions == ["read"]
        assert not tool.requires_approval


class TestToolRegistry:
    """工具注册表测试"""

    def setup_method(self):
        ToolRegistry.clear()

    def teardown_method(self):
        ToolRegistry.clear()

    def test_register_and_get(self):
        def echo(x):
            return x

        ToolRegistry.register("echo", echo, description="回显", permissions=["read"])
        tool = ToolRegistry.get("echo")
        assert tool is not None
        assert tool.name == "echo"
        assert tool.fn("hello") == "hello"

    def test_get_for_agent_with_permissions(self):
        ToolRegistry.register(
            "read_file", lambda: "data",
            description="读文件", permissions=["read_file"],
        )
        ToolRegistry.register(
            "write_file", lambda: None,
            description="写文件", permissions=["write_file"],
        )
        ToolRegistry.grant("reader", ["read_file"])

        tools = ToolRegistry.get_for_agent("reader")
        assert len(tools) == 1
        assert tools[0].name == "read_file"

    def test_unregistered_returns_none(self):
        assert ToolRegistry.get("nonexistent") is None

    def test_no_permissions_returns_empty(self):
        ToolRegistry.register("tool", lambda: None, permissions=["admin"])
        assert ToolRegistry.get_for_agent("guest") == []


class TestAgentRegistry:
    """Agent 注册表测试"""

    def setup_method(self):
        AgentRegistry.clear()

    def teardown_method(self):
        AgentRegistry.clear()

    def test_register_and_list(self):
        AgentRegistry.register(ChatAgent)
        AgentRegistry.register(RAGAgent)

        agents = AgentRegistry.list_agents()
        assert len(agents) == 2
        names = [a["name"] for a in agents]
        assert "chat_agent" in names
        assert "rag_agent" in names

    def test_get_existing(self):
        AgentRegistry.register(ChatAgent)
        entry = AgentRegistry.get("chat_agent")
        assert entry["name"] == "chat_agent"

    def test_get_nonexistent(self):
        assert AgentRegistry.get("bogus") is None

    def test_get_cls(self):
        AgentRegistry.register(ChatAgent)
        cls = AgentRegistry.get_cls("chat_agent")
        assert cls is ChatAgent


class TestApprovalManager:
    """审批管理器测试"""

    def setup_method(self):
        self.am = ApprovalManager()

    def test_request_lifecycle(self):
        rid = str(uuid.uuid4())
        req = self.am.request(rid, "delete", {"id": 1}, "测试删除", RiskLevel.HIGH)
        assert req.status == "pending"
        assert len(self.am.list_pending()) == 1

        assert self.am.approve(rid)
        assert self.am.get(rid).status == "approved"
        assert len(self.am.list_pending()) == 0

    def test_reject(self):
        rid = str(uuid.uuid4())
        self.am.request(rid, "write", {}, "测试", RiskLevel.MEDIUM)
        self.am.reject(rid, "不需要")
        assert self.am.get(rid).status == "rejected"

    def test_cleanup(self):
        rid = str(uuid.uuid4())
        self.am.request(rid, "op", {}, "", RiskLevel.LOW)
        self.am.cleanup(rid)
        assert self.am.get(rid) is None

    def test_invalid_action(self):
        rid = str(uuid.uuid4())
        self.am.request(rid, "op", {}, "", RiskLevel.LOW)
        self.am.approve(rid)
        # 重复审批应该失败
        assert not self.am.approve(rid)


class TestPromptManager:
    """Prompt 管理器测试"""

    def test_load_chat_agent_prompt(self):
        pm = PromptManager()
        template = pm.get("chat_agent")
        assert len(template) > 0
        assert "助手" in template or "AI" in template

    def test_variable_interpolation(self):
        pm = PromptManager()
        result = pm.get("supervisor", agent_list="- test_agent: 测试")
        assert "test_agent" in result

    def test_fallback_to_default(self):
        pm = PromptManager()
        # 不存在的 prompt 使用内置默认值
        template = pm.get("nonexistent_agent")
        assert template == ""

    def test_list_versions(self):
        pm = PromptManager()
        versions = pm.list_versions()
        # 至少应该有 supervisor 和 chat_agent
        assert "supervisor" in versions or "chat_agent" in versions


class TestChatAgent:
    """ChatAgent 测试"""

    def test_metadata(self):
        assert ChatAgent.AGENT_NAME == "chat_agent"
        assert len(ChatAgent.AGENT_DESCRIPTION) > 0

    def test_instantiation(self):
        mock_llm = MagicMock()
        agent = ChatAgent(model=mock_llm)
        assert agent.name == "chat_agent"
        assert agent.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_invoke_with_mock_llm(self):
        """测试 Agent 调用 LLM 的基本流程"""
        from langchain_core.messages import AIMessage

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="你好，有什么可以帮你的？")

        agent = ChatAgent(model=mock_llm)

        # 构建测试 State
        from langchain_core.messages import HumanMessage
        state = {
            "session_id": "test",
            "messages": [HumanMessage(content="你好")],
        }

        result = await agent(state)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "有什么可以帮你的" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_agent_status_transitions(self):
        mock_llm = AsyncMock()
        from langchain_core.messages import AIMessage
        mock_llm.ainvoke.return_value = AIMessage(content="OK")

        agent = ChatAgent(model=mock_llm)
        assert agent.status == AgentStatus.IDLE

        from langchain_core.messages import HumanMessage
        await agent({"session_id": "t", "messages": [HumanMessage(content="hi")]})
        assert agent.status == AgentStatus.DONE


class TestRAGAgent:
    """RAGAgent 测试"""

    def test_metadata(self):
        assert RAGAgent.AGENT_NAME == "rag_agent"
        assert len(RAGAgent.AGENT_DESCRIPTION) > 0

    def test_tools_property(self):
        mock_llm = MagicMock()
        # 不带搜索工具的 RAG Agent
        agent = RAGAgent(model=mock_llm)
        assert agent.tools == []

        # 带搜索工具
        search_tool = MagicMock()
        agent_with_search = RAGAgent(model=mock_llm, search_tool=search_tool)
        assert len(agent_with_search.tools) == 1


class TestMultiAgentState:
    """MultiAgentState 结构测试"""

    def test_minimal_state(self):
        from langchain_core.messages import HumanMessage
        state = {
            "session_id": "s1",
            "messages": [HumanMessage(content="hello")],
        }
        assert state["session_id"] == "s1"
        assert len(state["messages"]) == 1

    def test_extended_state(self):
        from langchain_core.messages import HumanMessage
        state = {
            "session_id": "s1",
            "messages": [HumanMessage(content="hello")],
            "current_agent": "chat_agent",
            "next_agent": None,
            "agent_outputs": {"chat_agent": {"content": "hi"}},
            "pending_approval": None,
        }
        assert state["current_agent"] == "chat_agent"
        assert state["pending_approval"] is None
