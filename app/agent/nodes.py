"""LangGraph 节点定义 — 每个节点对应一个 Agent 或编排逻辑"""
import json
import re

from langchain_core.messages import SystemMessage

from app.agent.state import MultiAgentState
from app.agent.base import BaseAgent
from app.agent.registry import AgentRegistry
from app.agent.prompts.manager import get_prompt_manager
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
    """Supervisor 节点 — 分析意图并决策路由

    动态从 AgentRegistry 获取可用 Agent 列表，
    使用 LLM 分析用户意图，输出 next_agent 决策。
    """
    logger.info("[Supervisor] 开始分析意图")

    # 动态获取已注册的 Agent 列表
    agents = AgentRegistry.list_agents()
    agent_list_str = "\n".join(
        f"- {a['name']}: {a['description']}" for a in agents
    )

    # 加载 supervisor prompt（PromptManager 自动处理回退）
    prompt_manager = get_prompt_manager()
    system_prompt = prompt_manager.get("supervisor", agent_list=agent_list_str)

    # 构建完整上下文（不污染原始 messages）
    full_messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    try:
        result = await dispatcher.think(full_messages)
        content = result["message"].content

        decision = _extract_json(content)
        next_agent = decision.get("next_agent")
        reason = decision.get("reason", "")

        # 验证路由目标是否合法
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
            "next_agent": "chat_agent",     # 出错时回退到通用对话 Agent
            "agent_outputs": {"supervisor": {"error": str(e), "fallback": "chat_agent"}},
        }


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON 对象"""
    # 匹配 ```json ... ``` 代码块
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        return json.loads(match.group(1))
    # 匹配裸 JSON 对象
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return json.loads(match.group(0))
    return {}


# ── 路由函数 ─────────────────────────────────────────────

def route_after_supervisor(state: MultiAgentState) -> str:
    """根据 Supervisor 的决策路由到下一个节点

    动态匹配已注册 Agent 名称，未知 Agent 或 null → finalize。
    """
    next_agent = state.get("next_agent")
    if next_agent and AgentRegistry.get(next_agent) is not None:
        return next_agent
    return "finalize"
