import logging

from fastapi import APIRouter, HTTPException

from app.core.llm import qwen
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

        messages = MessageAdapter.to_langchain(req.messages)

        res = llm.invoke(messages)
        print(f"LLM Response: {res}")
        content = res.content
        print(f"Content: {content}")

        return {
            "code": 0,
            "data": content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
