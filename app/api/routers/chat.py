import logging

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import qwen
from app.core.redis_manager import redis_client
from app.schemas.chat import ChatReq
from app.tools.message_adapter import MessageAdapter

router = APIRouter(prefix="/chat")


@router.post("/llm")
async def chat(req: ChatReq):
    print("Chat Request:", req.model_dump_json())
    try:
        llm = qwen(
            temperature=req.temperature
        )

        # 从Redis中获取会话历史
        session_history = await redis_client.get_object(req.session_id)
        if session_history is not None:
            session_history.append(HumanMessage(content=req.message))
        else:
            session_history = [
                SystemMessage(content="你是一个严谨的助手。"),
                HumanMessage(content=req.message)
            ]

        messages = MessageAdapter.to_langchain(session_history)

        res = llm.invoke(messages)
        print(f"LLM Response: {res}")
        content = res.content

        # 保存会话历史到Redis
        await redis_client.set_object(req.session_id, session_history)
        print(f"Content: {content}")

        return {
            "code": 0,
            "data": content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
