import redis.asyncio as redis  # 建议安装 redis 库: pip install redis
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import StateGraph
from app.agents.nodes import chat_node
from app.agents.state import State
from app.core.config import settings


def build_graph(checkpointer):
    builder = StateGraph(State)
    builder.add_node("chat", chat_node)
    builder.set_entry_point("chat")
    builder.set_finish_point("chat")
    # 记忆必须在编译时绑定
    return builder.compile(checkpointer=checkpointer)


class GraphRuntime:
    def __init__(self):
        self.graph = None
        self._saver = None

    async def _init(self):
        """初始化逻辑：显式管理连接池"""
        if self.graph is None:
            url = f"redis://{settings.redis.host}:{settings.redis.port}"

            # 实例化 Saver
            self._saver = AsyncRedisSaver(url)

            # 编译图
            self.graph = build_graph(self._saver)

    async def invoke(self, session_id: str, message: str):
        await self._init()

        # thread_id 是 Redis 中区分不同对话的唯一键
        config = {"configurable": {"thread_id": session_id}}
        state = {"messages": [HumanMessage(content=message)]}

        # ainvoke 会自动通过 self._saver 去 Redis 读写状态
        result = await self.graph.ainvoke(state, config=config)

        if "messages" in result and result["messages"]:
            return result["messages"][-1].content
        return ""
