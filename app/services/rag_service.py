from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.postgre_manager import get_db


class RagService:

    def __init__(self, db: Session):
        self.db = db


def get_rag_service(db: Session = Depends(get_db)):
    return RagService(db)
