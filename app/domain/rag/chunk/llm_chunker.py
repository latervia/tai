import inspect
import json
import re

from app.shared.logger import logger
from app.infrastructure.llm import qwen
from app.domain.rag.chunk.base import BaseChunker
from app.domain.rag.chunk.models import Chunk


class LlmChunker(BaseChunker):

    def __init__(self, max_chunk_length: int = 1000):
        self.llm = qwen()
        self.max_chunk_length = max_chunk_length

    def chunk(self, text: str) -> list[Chunk]:
        segments = self._split_by_llm(text)
        return [
            Chunk(content=seg, chunk_index=i)
            for i, seg in enumerate(segments)
        ]

    def _split_by_llm(self, text: str) -> list[str]:
        prompt = inspect.cleandoc(f"""
            根据输入文本的特性，请对输入文本进行切分，要求如下：
            1. 每个切分后的文本长度不超过{self.max_chunk_length}个字符。
            2. 每个切分后的文本应该是一个完整的句子。
            3. 返回一个列表，列表中的每个元素都是一个字符串
            4. 严格按照["字符串1", "字符串2", "字符串3"]的格式返回，不要返回任何多余内容
        """)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]

        response = self.llm.invoke(messages)
        content = response.content

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("LLM 返回的不是标准 JSON，尝试正则提取")
            match = re.search(r"\[.*]", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.error(f"无法解析 LLM 切分结果: {content[:500]}")
            return [text]
