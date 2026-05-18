"""长期记忆管理器 — 基于 Milvus 的 Agent 记忆存储与检索"""
import time
from typing import Optional

from app.domain.agent.states.state import MultiAgentState
from app.shared.logger import logger


class MemoryManager:
    """Agent 长期记忆管理器

    管理记忆的生命周期：
    - 存储：重要信息写入 Milvus 向量库
    - 检索：根据当前上下文查询相关历史记忆
    - 衰减：按时间或重要性衰减/清理旧记忆

    当前为结构化接口，具体 embedding 和 Milvus 调用由 RAG 层实现。
    """

    def __init__(self, embedding_fn=None, vector_store=None):
        """
        Args:
            embedding_fn: 文本 → 向量的嵌入函数
            vector_store: Milvus 或其他向量存储的实例
        """
        self._embedding = embedding_fn
        self._store = vector_store

    # ── 写入 ──────────────────────────────────────────────

    async def save_context(
        self,
        session_id: str,
        content: str,
        *,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ):
        """将重要上下文存入长期记忆

        Args:
            session_id: 会话标识
            content: 要存储的文本内容
            importance: 重要性评分 0~1（越高越容易被检索到）
            metadata: 附加元数据（来源、时间等）
        """
        if self._store is None:
            logger.debug("[Memory] 向量存储未配置，跳过写入")
            return

        doc = {
            "session_id": session_id,
            "content": content,
            "importance": importance,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        # embedding → vector_store 的实际写入由 RAG pipeline 处理
        logger.info(f"[Memory] 保存记忆: session={session_id}, importance={importance}")

    # ── 检索 ──────────────────────────────────────────────

    async def recall(
        self,
        session_id: str,
        query: str,
        *,
        top_k: int = 5,
        min_importance: float = 0.0,
    ) -> list[dict]:
        """检索与当前查询相关的历史记忆

        Args:
            session_id: 会话标识（可跨 session 检索）
            query: 当前查询文本
            top_k: 返回条数
            min_importance: 最低重要性阈值

        Returns:
            [{"content": "...", "score": 0.95, "metadata": {...}}, ...]
        """
        if self._store is None:
            return []

        # 实际检索由 RAG pipeline 实现
        logger.info(f"[Memory] 检索记忆: query='{query[:50]}...', top_k={top_k}")
        return []

    # ── 自动提取 ──────────────────────────────────────────

    async def extract_from_state(self, state: MultiAgentState):
        """从当前 State 中自动提取值得长期记忆的内容

        规则：
        1. 摘要记忆直接存储
        2. 用户的明确偏好/信息（"我叫XX"）标记为高重要性
        3. Agent 产出的结论性内容存储
        """
        summary = state.get("summary_memory")
        if summary:
            session_id = state.get("session_id", "unknown")
            await self.save_context(session_id, summary, importance=0.8)

    # ── 维护 ──────────────────────────────────────────────

    async def decay(self, session_id: str, older_than_days: int = 30):
        """衰减或清理旧记忆"""
        logger.info(f"[Memory] 清理 {session_id} 中超过 {older_than_days} 天的记忆")
        # 降低旧记忆的 importance 分值，而非直接删除
