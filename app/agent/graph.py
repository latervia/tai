"""Multi-Agent Graph — 纯图构建 + 运行时生命周期"""
from typing import Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import StateGraph, END

from app.agent.base import BaseAgent
from app.agent.registry import AgentRegistry, bootstrap_agents
from app.agent.state import MultiAgentState
from app.agent.nodes import (
    create_agent_node,
    supervisor_node,
    finalize_node,
    route_after_supervisor,
)
from app.core.config import settings
from app.core.logger import logger
from app.core.model.model_dispatcher import ModelDispatcher
from app.core.model.model_factory import ModelProvider


# ── 图构建 ───────────────────────────────────────────────

def build_multi_agent_graph(
    agent_nodes: dict[str, callable],
    supervisor,
    finalize,
    checkpointer,
):
    """构建 Multi-Agent 状态图

    图结构:
        START → supervisor → [agent_1 | agent_2 | ... | finalize]
                                 ↓
                             finalize → END
    """
    builder = StateGraph(MultiAgentState)

    builder.add_node("supervisor", supervisor)
    for name, node_fn in agent_nodes.items():
        builder.add_node(name, node_fn)
    builder.add_node("finalize", finalize)

    builder.set_entry_point("supervisor")

    route_map = {name: name for name in agent_nodes}
    route_map["finalize"] = "finalize"
    builder.add_conditional_edges("supervisor", route_after_supervisor, route_map)

    for name in agent_nodes:
        builder.add_edge(name, "finalize")

    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


# ── 运行时 ───────────────────────────────────────────────

class GraphRuntime:
    """Multi-Agent 图运行时

    管理 Agent 实例、ModelDispatcher 和 LangGraph 的完整生命周期。
    """

    def __init__(self):
        self.graph = None
        self._saver = None
        self._dispatcher: Optional[ModelDispatcher] = None
        self._agents: dict[str, BaseAgent] = {}

    async def _init(self):
        if self.graph is not None:
            return

        url = f"redis://{settings.redis.host}:{settings.redis.port}"
        self._saver = AsyncRedisSaver(url)

        self._dispatcher = ModelDispatcher(
            ModelProvider.QWEN,
            max_retries=2,
            token_budget=None,
        )

        # 从注册表动态创建所有 Agent 实例（首次调用时自动 bootstrap）
        bootstrap_agents()
        for entry in AgentRegistry.list_agents():
            agent_name = entry["name"]
            agent_cls = AgentRegistry.get_cls(agent_name)
            if agent_cls:
                agent = agent_cls(model=self._dispatcher.primary_llm)
                self._agents[agent_name] = agent
                logger.info(f"[GraphRuntime] 实例化 Agent: {agent_name}")

        # 包装为 LangGraph 节点
        agent_nodes = {
            name: create_agent_node(agent)
            for name, agent in self._agents.items()
        }

        # Supervisor（dispatcher 通过闭包绑定，与普通 Agent 统一接口）
        _dispatcher = self._dispatcher

        async def supervisor(state: MultiAgentState) -> dict:
            return await supervisor_node(state, _dispatcher)

        self.graph = build_multi_agent_graph(
            agent_nodes=agent_nodes,
            supervisor=supervisor,
            finalize=finalize_node,
            checkpointer=self._saver,
        )
        logger.info(f"[GraphRuntime] Multi-Agent 图已编译 ({len(agent_nodes)} agents)")

    async def invoke(self, session_id: str, message: str) -> str:
        await self._init()

        # → tracing: 标记请求开始
        from app.core.deps import get_trace_collector, get_cost_controller
        get_trace_collector().start_request(session_id, message)
        get_cost_controller().start_session(session_id)

        config = {"configurable": {"thread_id": session_id}}
        state = {"session_id": session_id, "messages": [HumanMessage(content=message)]}

        result = await self.graph.ainvoke(state, config=config)

        # → tracing: 标记请求结束
        get_trace_collector().finish_request(session_id)

        messages = result.get("messages", [])
        return messages[-1].content if messages else ""

    @property
    def dispatcher(self) -> ModelDispatcher:
        return self._dispatcher

    @property
    def agents(self) -> dict[str, BaseAgent]:
        return dict(self._agents)
