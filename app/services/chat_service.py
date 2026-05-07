from uuid6 import uuid7

from app.agents.graph import GraphRuntime, build_graph


class ChatService:

    def __init__(self, runtime: GraphRuntime):
        self.runtime = runtime

    async def chat(self, session_id: str, message: str):
        if session_id is None:
            session_id = str(uuid7())

        return await self.runtime.invoke(session_id, message)


def get_chat_service():
    runtime = GraphRuntime()
    return ChatService(runtime)
