"""路由与编排集成测试"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage

from app.agent.nodes import (
    route_after_supervisor,
    _extract_json,
    create_agent_node,
)
from app.agent.registry import AgentRegistry
from app.agent.workers.chat_agent import ChatAgent
from app.agent.workers.rag_agent import RAGAgent
from app.agent.graph import _bootstrap_agents


class TestJSONExtraction:
    """JSON 提取逻辑测试"""

    def test_plain_json(self):
        result = _extract_json('{"next_agent": "chat_agent", "reason": "test"}')
        assert result["next_agent"] == "chat_agent"

    def test_code_block_json(self):
        result = _extract_json('```json\n{"next_agent": null}\n```')
        assert result["next_agent"] is None

    def test_code_block_no_lang(self):
        result = _extract_json('```\n{"next_agent": "rag_agent"}\n```')
        assert result["next_agent"] == "rag_agent"

    def test_embedded_json(self):
        result = _extract_json('分析结果：{"next_agent": "chat_agent", "reason": "简单对话"}')
        assert result["next_agent"] == "chat_agent"

    def test_no_json(self):
        result = _extract_json("没有 JSON 的文本")
        assert result == {}


class TestRouting:
    """路由逻辑测试"""

    def setup_method(self):
        AgentRegistry.clear()
        _bootstrap_agents()

    def teardown_method(self):
        AgentRegistry.clear()

    def test_route_to_chat_agent(self):
        state = {"messages": [], "next_agent": "chat_agent"}
        assert route_after_supervisor(state) == "chat_agent"

    def test_route_to_rag_agent(self):
        state = {"messages": [], "next_agent": "rag_agent"}
        assert route_after_supervisor(state) == "rag_agent"

    def test_route_null_to_finalize(self):
        state = {"messages": [], "next_agent": None}
        assert route_after_supervisor(state) == "finalize"

    def test_route_unknown_to_finalize(self):
        state = {"messages": [], "next_agent": "unknown_agent"}
        assert route_after_supervisor(state) == "finalize"

    def test_route_missing_key_to_finalize(self):
        state = {"messages": []}
        assert route_after_supervisor(state) == "finalize"


class TestCreateAgentNode:
    """Agent 节点工厂测试"""

    @pytest.mark.asyncio
    async def test_agent_node_wrapping(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="test response")

        agent = ChatAgent(model=mock_llm)
        node_fn = create_agent_node(agent)

        state = {
            "session_id": "test",
            "messages": [HumanMessage(content="hello")],
        }

        result = await node_fn(state)
        assert "current_agent" in result
        assert result["current_agent"] == "chat_agent"
        assert "messages" in result


class TestAgentRegistryRouting:
    """Agent 注册表与路由的联动测试"""

    def setup_method(self):
        AgentRegistry.clear()

    def teardown_method(self):
        AgentRegistry.clear()

    def test_dynamic_agent_registration(self):
        """验证新增 Agent 后路由能正确识别"""
        AgentRegistry.register(ChatAgent)
        AgentRegistry.register(RAGAgent)

        assert AgentRegistry.get("chat_agent") is not None
        assert AgentRegistry.get("rag_agent") is not None
        assert AgentRegistry.get("nonexistent") is None

        # 验证路由匹配
        state = {"messages": [], "next_agent": "rag_agent"}
        assert route_after_supervisor(state) == "rag_agent"

    def test_bootstrap_registers_expected_agents(self):
        _bootstrap_agents()
        agents = AgentRegistry.list_agents()
        agent_names = [a["name"] for a in agents]
        assert "chat_agent" in agent_names
        assert "rag_agent" in agent_names
