import inspect
import json

from app.core.llm import qwen
from app.rag.chunk.base import BaseChunker


class LlmChunker(BaseChunker):

    def __init__(self):
        self.llm = qwen()

    def chunk(self, text: str):
        self.chunk_with_llm(text)

    def chunk_with_llm(self, text: str, prompt: str | None = None) -> list[str]:
        default_prompt = inspect.cleandoc("""
        根据输入文本的特性，请对输入文本进行切分，要求如下：
        1. 每个切分后的文本长度不超过1000个字符。
        2. 每个切分后的文本应该是一个完整的句子。
        3. 返回一个列表，列表中的每个元素都是一个字符串
        4. 严格按照["字符串1", "字符串2", "字符串3"]的格式返回，不要返回任何多余内容
        """)

        if not prompt:
            prompt = default_prompt

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]

        response = self.llm.invoke(messages)
        content = response.content
        json_content = json.loads(content)
        return json_content
