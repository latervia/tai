"""Multi-Agent Graph 构建 — Supervisor + Worker Agents + Finalize"""
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import StateGraph, END

from app.agent.base import BaseAgent
from app.agent.registry import AgentRegistry
from app.agent.state import MultiAgentState
from app.agent.nodes import (
    create_agent_node,
    supervisor_node,
    route_after_supervisor,
)
from app.agent.workers.chat_agent import ChatAgent
from app.agent.workers.rag_agent import RAGAgent
from app.core.config import settings
from app.core.logger import logger
from app.core.model.model_dispatcher import ModelDispatcher
from app.core.model.model_factory import ModelProvider


# ── Agent 注册引导 ───────────────────────────────────────

def _bootstrap_agents():
    """启动时自动注册所有已知的 Agent 类"""
    if not AgentRegistry.list_agents():
        AgentRegistry.register(ChatAgent)
        AgentRegistry.register(RAGAgent)
        logger.info(f"[Bootstrap] 已注册 Agent: {[a['name'] for a in AgentRegistry.list_agents()]}")


# ── Finalize 节点 ────────────────────────────────────────

async def finalize_node(state: MultiAgentState) -> dict:
    """汇总节点 — 在所有 Agent 执行完毕后生成最终回复

    如果已有 Agent 输出了回复消息，直接透传；
    否则基于 agent_outputs 汇总生成兜底回复。
    """
    logger.info("[Finalize] 汇总结果")

    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and last_msg.content:
            logger.info("[Finalize] 已有 Agent 回复，直接透传")
            return {}

    # 没有 AI 回复时，汇总 agent_outputs
    outputs = state.get("agent_outputs", {})
    parts = []
    for agent_name, output in outputs.items():
        if agent_name == "supervisor":
            continue    # supervisor 的决策信息不展示给用户
        if isinstance(output, dict) and output.get("content"):
            parts.append(output["content"])
        elif isinstance(output, dict) and output.get("status") == "failed":
            parts.append(f"[{agent_name}] 处理失败")

    if parts:
        return {"messages": [AIMessage(content="\n\n".join(parts))]}
    return {"messages": [AIMessage(content="抱歉，我暂时无法处理这个请求。")]}


# ── Graph 构建 ───────────────────────────────────────────

def build_multi_agent_graph(
    agent_nodes: dict[str, callable],
    supervisor,
    finalize,
    checkpointer,
):
    """构建 Multi-Agent 状态图

    图结构:
        START → supervisor → [agent_1 | agent_2 | ... | finalize]
                  ↑              ↓
                  └──────────────┘

    所有 Agent 节点由 agent_nodes 动态注入，
    路由规则由 route_after_supervisor + AgentRegistry 决定。
    """
    builder = StateGraph(MultiAgentState)

    # 注册所有 Agent 节点
    builder.add_node("supervisor", supervisor)
    for name, node_fn in agent_nodes.items():
        builder.add_node(name, node_fn)
    builder.add_node("finalize", finalize)

    builder.set_entry_point("supervisor")

    # Supervisor → 条件路由（动态生成路由映射）
    route_map = {name: name for name in agent_nodes}
    route_map["finalize"] = "finalize"
    builder.add_conditional_edges("supervisor", route_after_supervisor, route_map)

    # 各 Agent → Finalize
    for name in agent_nodes:
        builder.add_edge(name, "finalize")

    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


# ── 运行时 ───────────────────────────────────────────────

class GraphRuntime:
    """Multi-Agent 图运行时

    管理 Agent 实例、ModelDispatcher 和 LangGraph 的完整生命周期。
    保持与原 ChatService 的接口兼容。
    """

    def __init__(self):
        self.graph = None
        self._saver = None
        self._dispatcher: Optional[ModelDispatcher] = None
        self._agents: dict[str, BaseAgent] = {}

    async def _init(self):
        """延迟初始化 — 在第一次调用时建立连接和编译图"""
        if self.graph is not None:
            return

        # 0. 注册 Agent 类到全局注册表
        _bootstrap_agents()

        url = f"redis://{settings.redis.host}:{settings.redis.port}"

        # 1. Redis 检查点存储器
        self._saver = AsyncRedisSaver(url)

        # 2. 模型调度器
        self._dispatcher = ModelDispatcher(
            ModelProvider.QWEN,
            max_retries=2,
            token_budget=None,
        )

        # 3. 创建 Agent 实例（所有已注册的 Agent 类型）
        for entry in AgentRegistry.list_agents():
            agent_name = entry["name"]
            agent_cls = AgentRegistry.get_cls(agent_name)
            if agent_cls:
                agent = agent_cls(model=self._dispatcher.primary_llm)
                self._agents[agent_name] = agent
                logger.info(f"[GraphRuntime] 实例化 Agent: {agent_name}")

        # 4. 包装为 LangGraph 节点
        agent_nodes = {
            name: create_agent_node(agent)
            for name, agent in self._agents.items()
        }

        # 5. 创建 Supervisor 节点（闭包绑定 dispatcher）
        _dispatcher = self._dispatcher

        async def supervisor(state: MultiAgentState) -> dict:
            return await supervisor_node(state, _dispatcher)

        # 6. 编译图
        self.graph = build_multi_agent_graph(
            agent_nodes=agent_nodes,
            supervisor=supervisor,
            finalize=finalize_node,
            checkpointer=self._saver,
        )
        logger.info(f"[GraphRuntime] Multi-Agent 图已编译 ({len(agent_nodes)} agents)")

    async def invoke(self, session_id: str, message: str) -> str:
        """执行一次对话

        Args:
            session_id: 会话标识（用作 Redis 的 thread_id）
            message: 用户输入

        Returns:
            Agent 的最终回复文本
        """
        await self._init()

        config = {"configurable": {"thread_id": session_id}}
        state = {"session_id": session_id, "messages": [HumanMessage(content=message)]}

        result = await self.graph.ainvoke(state, config=config)

        messages = result.get("messages", [])
        if messages:
            return messages[-1].content
        return ""

    @property
    def dispatcher(self) -> ModelDispatcher:
        return self._dispatcher

    @property
    def agents(self) -> dict[str, BaseAgent]:
        """返回所有 Agent 实例（供测试或调试用）"""
        return dict(self._agents)
