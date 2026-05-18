from fastapi import APIRouter, Depends

from app.shared.logger import logger
from app.delivery.fastapi.schemas.chat import ChatReq
from app.services.chat_service import get_chat_service, ChatService

router = APIRouter(prefix="/chat")


@router.post("/llm")
async def chat(
        req: ChatReq,
        service: ChatService = Depends(get_chat_service)
):
    res = await service.chat(req.session_id, req.message)
    logger.info(f"chat res: {res}")

    return {
        "code": 0,
        "data": res,
        "message": "success"
    }
