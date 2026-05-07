from langchain_core.messages import HumanMessage
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import StateGraph
from app.agents.nodes import chat_node
from app.agents.state import State
from app.core.config import settings


# 1. 修改编译函数，接受 checkpointer 参数
def build_graph(checkpointer):
    builder = StateGraph(State)
    builder.add_node("chat", chat_node)
    builder.set_entry_point("chat")
    builder.set_finish_point("chat")

    # 必须在这里绑定 checkpointer，图才知道如何读写记忆
    return builder.compile(checkpointer=checkpointer)


class GraphRuntime:
    def __init__(self):
        self.graph = None
        self._saver_cm = None  # 存储上下文管理器
        self._saver = None  # 存储激活后的 saver 实例

    async def _init(self):
        if self.graph is None:
            url = f"redis://{settings.redis.host}:{settings.redis.port}"
            # 2. 获取上下文管理器
            self._saver_cm = AsyncRedisSaver.from_conn_string(url)
            # 3. 手动激活上下文，获取真正的 saver 实例
            # 注意：在长生命周期应用中，这种方式需要手动管理 __aenter__
            self._saver = await self._saver_cm.__aenter__()

            # 4. 编译时传入 saver
            self.graph = build_graph(self._saver)

    async def invoke(self, session_id: str, message: str):
        await self._init()

        # 5. 只需 thread_id，LangGraph 会自动根据它去 Redis 找历史记录
        config = {"configurable": {"thread_id": session_id}}
        state = {"messages": [HumanMessage(content=message)]}

        # 直接调用，无需再 async with
        result = await self.graph.ainvoke(state, config=config)

        if "messages" in result and result["messages"]:
            return result["messages"][-1].content
        return ""

    async def stop(self):
        """应用关闭时断开 Redis 连接"""
        if self._saver_cm:
            await self._saver_cm.__aexit__(None, None, None)