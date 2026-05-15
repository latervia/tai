"""LangGraph 节点定义 — 所有节点函数 + 路由逻辑"""
import json
import re

from langchain_core.messages import SystemMessage, AIMessage

from app.agent.state import MultiAgentState
from app.agent.base import BaseAgent
from app.agent.registry import AgentRegistry
from app.core.deps import get_prompt_manager
from app.core.logger import logger


# ── 节点工厂 ─────────────────────────────────────────────

def create_agent_node(agent: BaseAgent):
    """将一个 BaseAgent 实例包装为 LangGraph 节点函数"""

    async def node(state: MultiAgentState) -> dict:
        logger.info(f"[Node] 进入 {agent.name} 节点")
        result = await agent(state)
        result["current_agent"] = agent.name
        return result

    return node


# ── Supervisor 节点 ──────────────────────────────────────

async def supervisor_node(state: MultiAgentState, dispatcher) -> dict:
    """Supervisor — 分析意图并决策路由"""
    logger.info("[Supervisor] 开始分析意图")
    session_id = state.get("session_id", "")

    agents = AgentRegistry.list_agents()
    agent_list_str = "\n".join(
        f"- {a['name']}: {a['description']}" for a in agents
    )

    system_prompt = get_prompt_manager().get("supervisor", agent_list=agent_list_str)
    full_messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    try:
        result = await dispatcher.think(
            full_messages, session_id=session_id, agent_name="supervisor",
        )
        content = result["message"].content
        decision = _extract_json(content)
        next_agent = decision.get("next_agent")
        reason = decision.get("reason", "")

        if next_agent and AgentRegistry.get(next_agent) is None:
            logger.warning(f"[Supervisor] 路由到未知 Agent: {next_agent}，回退到 null")
            next_agent = None

        logger.info(f"[Supervisor] 路由决策: next_agent={next_agent}, reason={reason}")

        return {
            "next_agent": next_agent,
            "agent_outputs": {"supervisor": {"next_agent": next_agent, "reason": reason}},
        }
    except Exception as e:
        logger.error(f"[Supervisor] 路由分析失败: {e}")
        return {
            "next_agent": "chat_agent",
            "agent_outputs": {"supervisor": {"error": str(e), "fallback": "chat_agent"}},
        }


def _extract_json(text: str) -> dict:
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        return json.loads(match.group(1))
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return json.loads(match.group(0))
    return {}


# ── Finalize 节点 ────────────────────────────────────────

async def finalize_node(state: MultiAgentState) -> dict:
    """汇总节点 — 透传 Agent 回复或生成兜底"""
    logger.info("[Finalize] 汇总结果")

    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and last_msg.content:
            return {}

    outputs = state.get("agent_outputs", {})
    parts = []
    for agent_name, output in outputs.items():
        if agent_name == "supervisor":
            continue
        if isinstance(output, dict) and output.get("content"):
            parts.append(output["content"])
        elif isinstance(output, dict) and output.get("status") == "failed":
            parts.append(f"[{agent_name}] 处理失败")

    if parts:
        return {"messages": [AIMessage(content="\n\n".join(parts))]}
    return {"messages": [AIMessage(content="抱歉，我暂时无法处理这个请求。")]}


# ── 路由函数 ─────────────────────────────────────────────

def route_after_supervisor(state: MultiAgentState) -> str:
    """Supervisor 决策 → LangGraph 节点名称"""
    next_agent = state.get("next_agent")
    if next_agent and AgentRegistry.get(next_agent) is not None:
        return next_agent
    return "finalize"
