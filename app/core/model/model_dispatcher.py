"""模型调度器 — LLM 调用的统一入口，含限流、重试、追踪"""
import time
import asyncio
from typing import Optional

from langchain_core.messages import BaseMessage

from app.core.logger import logger
from app.core.model.model_factory import ModelProvider, ModelFactory


class BudgetExceededError(Exception):
    """Token 预算超限异常"""

    def __init__(self, used: int, limit: int):
        self.used = used
        self.limit = limit
        super().__init__(f"Token 预算超限: 已用 {used}/{limit}")


class ModelDispatcher:
    """统一的 LLM 调用调度器

    职责：
    1. 主备模型切换（primary → fallback）
    2. 指数退避重试（仅 5xx / 429）
    3. Token 消耗追踪与预算控制
    4. 结构化日志（为 tracing 面板提供数据）
    """

    def __init__(
        self,
        primary_provider: ModelProvider,
        *,
        max_retries: int = 2,
        token_budget: Optional[int] = None,    # None = 不限
        base_delay: float = 1.0,               # 重试基础延迟（秒）
    ):
        self.primary_llm = ModelFactory.get_model(primary_provider)
        self.fallback_llm = ModelFactory.get_model(ModelProvider.OLLAMA)
        self.max_retries = max_retries
        self.token_budget = token_budget
        self.base_delay = base_delay

        # 追踪状态
        self._tokens_used: int = 0             # 当前会话已消耗的 token
        self._call_count: int = 0              # LLM 调用次数
        self._call_records: list[dict] = []    # 详细调用记录

    # ── 公共接口 ──────────────────────────────────────────

    async def invoke(self, messages: list[BaseMessage], **kwargs) -> dict:
        """调用 LLM，自动处理重试和 fallback

        Returns:
            {"message": AIMessage, "tokens": {...}, "duration_ms": ...}
        """
        return await self._invoke_with_retry(self.primary_llm, messages, **kwargs)

    async def think(self, messages: list[BaseMessage], **kwargs) -> dict:
        """轻量调用 — 用于 Supervisor 的路由判断等简单任务"""
        # 使用低 temperature 获得稳定输出
        kwargs.setdefault("temperature", 0.1)
        return await self.invoke(messages, **kwargs)

    # ── 内部实现 ──────────────────────────────────────────

    async def _invoke_with_retry(
        self, llm, messages: list[BaseMessage], **kwargs
    ) -> dict:
        """带重试的 LLM 调用，遇到不可重试错误时切换到 fallback"""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self._do_invoke(llm, messages, **kwargs)
            except Exception as e:
                last_error = e
                # 4xx 错误（参数错误等）不应重试
                if self._is_client_error(e):
                    logger.warning(f"LLM 客户端错误，不重试: {e}")
                    break
                # 5xx / 429 可重试
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"LLM 调用失败 (attempt {attempt+1})，{delay}s 后重试: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"LLM 重试耗尽: {e}")

        # 主模型全部失败 → 尝试 fallback
        if llm is not self.fallback_llm:
            logger.warning("切换到 fallback 模型")
            try:
                return await self._do_invoke(self.fallback_llm, messages, **kwargs)
            except Exception as e:
                last_error = e

        raise last_error or RuntimeError("LLM 调用失败且无 fallback")

    async def _do_invoke(
        self, llm, messages: list[BaseMessage], **kwargs
    ) -> dict:
        """单次 LLM 调用 + token 追踪"""
        start = time.time()

        # 预算检查
        if self.token_budget and self._tokens_used >= self.token_budget:
            raise BudgetExceededError(self._tokens_used, self.token_budget)

        response = await llm.ainvoke(messages, **kwargs)

        duration_ms = (time.time() - start) * 1000

        # 提取 token 信息（兼容不同 LLM 的返回格式）
        tokens = self._extract_tokens(response)
        self._tokens_used += tokens.get("total", 0)
        self._call_count += 1

        # 记录调用 trace
        record = {
            "call_no": self._call_count,
            "model": getattr(llm, "model_name", "unknown"),
            "tokens": tokens,
            "duration_ms": round(duration_ms, 1),
            "input_preview": str(messages[-1].content)[:200] if messages else "",
        }
        self._call_records.append(record)
        logger.info(
            f"[LLM #{self._call_count}] {record['model']} "
            f"tokens={tokens} latency={record['duration_ms']}ms"
        )

        return {
            "message": response,
            "tokens": tokens,
            "duration_ms": duration_ms,
        }

    # ── 辅助方法 ──────────────────────────────────────────

    @staticmethod
    def _extract_tokens(response) -> dict:
        """从 LLM 响应中提取 token 计数"""
        tokens = {"input": 0, "output": 0, "total": 0}
        meta = getattr(response, "usage_metadata", None) or {}
        if meta:
            tokens["input"] = meta.get("input_tokens", 0)
            tokens["output"] = meta.get("output_tokens", 0)
            tokens["total"] = tokens["input"] + tokens["output"]
        # 兼容某些模型把 token 信息放在 response_metadata
        rm = getattr(response, "response_metadata", {}) or {}
        if not tokens["total"] and "token_usage" in rm:
            tu = rm["token_usage"]
            tokens["input"] = tu.get("prompt_tokens", 0)
            tokens["output"] = tu.get("completion_tokens", 0)
            tokens["total"] = tokens["input"] + tokens["output"]
        return tokens

    @staticmethod
    def _is_client_error(exception: Exception) -> bool:
        """判断是否为 4xx 类客户端错误（不应重试）"""
        msg = str(exception).lower()
        # 检查 HTTP 状态码
        for code in ["400", "401", "402", "403", "404"]:
            if code in msg:
                return True
        # 检查常见客户端错误关键词
        client_keywords = ["invalid request", "invalid api key", "permission", "quota"]
        return any(kw in msg for kw in client_keywords)

    # ── 查询接口 ──────────────────────────────────────────

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    @property
    def call_count(self) -> int:
        return self._call_count

    def get_trace(self) -> list[dict]:
        """获取本次会话的调用追踪记录"""
        return list(self._call_records)

    def reset_budget(self):
        """重置 token 计数（新会话开始时调用）"""
        self._tokens_used = 0
        self._call_count = 0
        self._call_records.clear()
