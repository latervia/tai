from langchain_core.messages import HumanMessage
from langgraph.checkpoint.redis import RedisSaver, AsyncRedisSaver
from langgraph.graph import StateGraph

from app.agents.nodes import chat_node
from app.agents.state import State
from app.core.config import settings


def build_graph():
    builder = StateGraph(State)
    builder.add_node("chat", chat_node)

    builder.set_entry_point("chat")
    builder.set_finish_point("chat")

    return builder.compile()


class GraphRuntime:

    def __init__(self):
        self.graph = None
        self._saver = None

    async def _init(self):
        # 这种模式确保了连接只初始化一次，且在整个 Runtime 生命周期内有效
        if self.graph is None:
            url = f"redis://{settings.redis.host}:{settings.redis.port}"
            # 手动激活并获取真正的实例
            self._saver = AsyncRedisSaver.from_conn_string(url)
            self.graph = build_graph()

    async def invoke(
            self,
            session_id: str,
            message: str,
    ):
        await self._init()

        config = self._build_config(session_id)
        state = self._build_state(message)

        async with self._saver as saver:
            config["configurable"]["checkpointer"] = saver

            result = await self.graph.ainvoke(
                state,
                config=config
            )

            # 提取最后一条消息的内容
            if "messages" in result and len(result["messages"]) > 0:
                return result["messages"][-1].content

            return ""

    def _build_config(
            self,
            session_id: str,
    ):
        return {
            "configurable": {
                "thread_id": session_id
            }
        }

    def _build_state(self, message: str):
        return {
            "messages": [
                HumanMessage(content=message)
            ]
        }
