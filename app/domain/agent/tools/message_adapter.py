"""消息格式适配器 — LangChain 消息与自定义格式互转"""
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class MessageAdapter:
    ROLE_MAP = {
        "human": HumanMessage,
        "ai": AIMessage,
        "system": SystemMessage,
    }

    @classmethod
    def to_langchain(cls, messages):
        return [
            cls.ROLE_MAP[m.role](content=m.content)
            for m in messages
        ]

    @staticmethod
    def from_langchain(message):
        return {"role": "ai", "content": message.content}
