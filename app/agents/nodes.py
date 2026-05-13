from app.agents.state import State
from app.core.model import qwen


def chat_node(state: State):
    messages = state['messages']
    llm = qwen()
    res = llm.invoke(messages)
    return {
        "messages": messages + [res]
    }
