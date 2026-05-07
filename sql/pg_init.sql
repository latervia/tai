-- 1. 知识库主表
CREATE TABLE tbl_kb
(
    id                UUID PRIMARY KEY         DEFAULT gen_random_uuid(),
    name              VARCHAR(255) NOT NULL,
    description       TEXT,
    milvus_collection VARCHAR(255) NOT NULL,
    status            VARCHAR(20)              DEFAULT 'active',
    chunk_strategy    JSONB,
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. 文档管理表
CREATE TABLE tbl_document
(
    id         UUID PRIMARY KEY         DEFAULT gen_random_uuid(),
    kb_id      UUID, -- 逻辑关联 tbl_rag.id
    file_name  VARCHAR(500) NOT NULL,
    file_type  VARCHAR(50),
    file_size  BIGINT,
    doc_hash   VARCHAR(64),
    source_url TEXT,
    status     VARCHAR(20)              DEFAULT 'processing',
    error_msg  TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. 切片引用表
CREATE TABLE tbl_chunk
(
    id               UUID PRIMARY KEY         DEFAULT gen_random_uuid(),
    doc_id           UUID, -- 逻辑关联 tbl_document.id
    kb_id            UUID, -- 逻辑关联 tbl_rag.id
    milvus_entity_id BIGINT,
    content          TEXT NOT NULL,
    chunk_order      INT,
    token_count      INT,
    metadata         JSONB,
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 加速文档表的关联查询
CREATE INDEX idx_doc_kb_id ON tbl_document (kb_id);

-- 加速切片表的关联查询
CREATE INDEX idx_chunk_doc_id ON tbl_chunk (doc_id);
CREATE INDEX idx_chunk_kb_id ON tbl_chunk (kb_id);

-- 加速 Milvus 回查内容的查询
CREATE INDEX idx_chunk_milvus_id ON tbl_chunk (milvus_entity_id);