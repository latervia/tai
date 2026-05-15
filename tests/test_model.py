"""模型调度与成本控制测试"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.model.model_dispatcher import ModelDispatcher, BudgetExceededError
from app.core.model.model_factory import ModelProvider
from app.core.cost import CostController, CostReport


class TestModelDispatcher:
    """ModelDispatcher 测试 — 不实际调用 LLM"""

    @pytest.fixture
    def mock_llm(self):
        from langchain_core.messages import AIMessage

        llm = AsyncMock()
        response = AIMessage(content="mock response")
        # 模拟 usage_metadata
        response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        llm.ainvoke.return_value = response
        llm.model_name = "mock-model"
        return llm

    @pytest.mark.asyncio
    async def test_successful_invoke(self, mock_llm):
        with patch.object(ModelDispatcher, '_do_invoke', new_callable=AsyncMock) as mock_do:
            mock_do.return_value = {
                "message": mock_llm.ainvoke.return_value,
                "tokens": {"input": 10, "output": 5, "total": 15},
                "duration_ms": 100.0,
            }

            dispatcher = ModelDispatcher(ModelProvider.QWEN, max_retries=0)
            result = await dispatcher.invoke([])
            assert "message" in result
            assert result["tokens"]["total"] == 15

    def test_extract_tokens(self):
        class MockResponse:
            usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
            response_metadata = {}

        tokens = ModelDispatcher._extract_tokens(MockResponse())
        assert tokens["input"] == 100
        assert tokens["output"] == 50
        assert tokens["total"] == 150

    def test_extract_tokens_alt_format(self):
        class MockResponse:
            usage_metadata = {}
            response_metadata = {
                "token_usage": {"prompt_tokens": 200, "completion_tokens": 80}
            }

        tokens = ModelDispatcher._extract_tokens(MockResponse())
        assert tokens["input"] == 200
        assert tokens["output"] == 80
        assert tokens["total"] == 280

    def test_client_error_detection(self):
        assert ModelDispatcher._is_client_error(Exception("HTTP 400 Bad Request"))
        assert ModelDispatcher._is_client_error(Exception("Error code 401"))
        assert ModelDispatcher._is_client_error(Exception("Error 403 Forbidden"))
        assert ModelDispatcher._is_client_error(Exception("Invalid API key"))
        assert not ModelDispatcher._is_client_error(Exception("HTTP 500"))
        assert not ModelDispatcher._is_client_error(Exception("Connection timeout"))

    def test_token_tracking(self):
        dispatcher = ModelDispatcher(ModelProvider.QWEN)
        assert dispatcher.tokens_used == 0
        assert dispatcher.call_count == 0

    def test_reset_budget(self):
        dispatcher = ModelDispatcher(ModelProvider.QWEN)
        dispatcher._tokens_used = 1000
        dispatcher._call_count = 5
        dispatcher.reset_budget()
        assert dispatcher.tokens_used == 0
        assert dispatcher.call_count == 0


class TestCostController:
    """成本控制器测试"""

    def test_session_budget(self):
        cc = CostController(default_session_budget=1000)
        cc.start_session("session_1")
        assert cc.remaining("session_1") == 1000
        assert cc.check_budget("session_1")

    def test_token_tracking(self):
        cc = CostController(default_session_budget=500)
        cc.start_session("session_1")
        cc.track("session_1", {"total": 300})
        assert cc.remaining("session_1") == 200
        cc.track("session_1", {"total": 250})
        assert cc.remaining("session_1") == 0

    def test_budget_exceeded(self):
        cc = CostController(default_session_budget=100)
        cc.start_session("session_1")
        cc.track("session_1", {"total": 150})
        assert not cc.check_budget("session_1")
        assert cc.remaining("session_1") == 0

    def test_report(self):
        cc = CostController(default_session_budget=1000)
        cc.start_session("s1")
        cc.track("s1", {"total": 400})
        cc.track("s1", {"total": 200})

        report = cc.get_report("s1")
        assert report.total_tokens == 600
        assert report.llm_calls == 2
        assert report.budget_remaining == 400
        assert not report.is_over_budget

    def test_end_session(self):
        cc = CostController(default_session_budget=1000)
        cc.start_session("s1")
        cc.track("s1", {"total": 100})
        report = cc.end_session("s1")
        assert report.total_tokens == 100
        # 结束后会话应被清理
        assert cc.get_report("s1").total_tokens == 0

    def test_auto_start_session(self):
        cc = CostController(default_session_budget=1000)
        # 未显式 start 的会话自动创建
        cc.track("auto_session", {"total": 50})
        assert cc.remaining("auto_session") == 950

    def test_daily_total(self):
        cc = CostController(default_session_budget=1000)
        cc.track("s1", {"total": 100})
        cc.track("s2", {"total": 200})
        assert cc.daily_total == 300
