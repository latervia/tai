from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class MessageAdapter:
    ROLE_MAP = {
        "user": HumanMessage,
        "assistant": AIMessage,
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
        return {
            "role": "assistant",
            "content": message.content
        }
