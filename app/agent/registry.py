"""Agent 注册表 — 集中管理所有可用 Agent 的元数据"""
from typing import Optional

from app.agent.base import BaseAgent
from app.core.logger import logger


class AgentRegistry:
    """Agent 注册中心

    供 Supervisor 查询可用 Agent 列表，决定路由目标。
    启动时通过 bootstrap() 一次性注册所有已知 Agent。
    """
    _entries: dict[str, dict] = {}

    @classmethod
    def register(cls, agent_cls: type[BaseAgent]):
        name = agent_cls.AGENT_NAME
        cls._entries[name] = {
            "name": name,
            "description": agent_cls.AGENT_DESCRIPTION,
            "cls": agent_cls,
        }

    @classmethod
    def list_agents(cls) -> list[dict]:
        return [
            {"name": e["name"], "description": e["description"]}
            for e in cls._entries.values()
        ]

    @classmethod
    def get(cls, name: str) -> Optional[dict]:
        return cls._entries.get(name)

    @classmethod
    def get_cls(cls, name: str) -> Optional[type[BaseAgent]]:
        entry = cls._entries.get(name)
        return entry["cls"] if entry else None

    @classmethod
    def clear(cls):
        cls._entries.clear()


# ── 启动引导 ─────────────────────────────────────────────

def bootstrap_agents():
    """应用启动时调用 — 注册所有已知的 Agent 类"""
    if AgentRegistry.list_agents():
        return  # 已注册，跳过

    from app.agent.workers.chat_agent import ChatAgent
    from app.agent.workers.rag_agent import RAGAgent

    AgentRegistry.register(ChatAgent)
    AgentRegistry.register(RAGAgent)
    logger.info(f"[Bootstrap] 已注册 Agent: {[a['name'] for a in AgentRegistry.list_agents()]}")
