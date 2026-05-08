import uuid
from datetime import datetime
from typing import Optional, Any, Dict

from sqlalchemy import String, Text, BigInteger, Integer, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.postgre_manager import Base


class KBModel(Base):
    __tablename__ = "tbl_kb"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    milvus_collection: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), server_default="active")
    chunk_strategy: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )


# 2. 文档管理表
class DocumentModel(Base):
    __tablename__ = "tbl_document"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))  # 仅逻辑关联
    file_name: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[Optional[str]] = mapped_column(String(50))
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger)
    doc_hash: Mapped[Optional[str]] = mapped_column(String(64))
    source_url: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), server_default="processing")
    error_msg: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# 3. 切片引用表
class ChunkModel(Base):
    __tablename__ = "tbl_chunk"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))  # 仅逻辑关联
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))  # 仅逻辑关联
    milvus_entity_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_order: Mapped[Optional[int]] = mapped_column(Integer)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column("metadata", JSONB)  # metadata 是关键词，建议加下划线或映射
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
