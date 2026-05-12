import inspect
from abc import ABC, abstractmethod
from typing import Annotated, Literal

from app.core.llm import ollama, qwen


class BaseConvertor(ABC):

    def __init__(self, way: str):
        self.way = way
        self.vlm = ollama()
        self.llm = qwen()

    @abstractmethod
    def convert(
            self,
            doc_id: str,
            source_type: Annotated[Literal["pdf", "docx"], "源文件格式"] = "pdf",
            target_type: Annotated[Literal["txt", "html", "md"], "目标文件格式"] = "md"
    ):
        pass

    @classmethod
    def get_source_name(
            cls,
            doc_id: str,
            source_type: str
    ):
        return f"{doc_id}/source.{source_type}"

    def get_target_name(
            self,
            doc_id: str,
            target_type: str
    ):
        return f"{doc_id}/target_by_{self.way}.{target_type}"

    def get_assets_dir(self, doc_id: str, ):
        return f"{doc_id}/assets_{self.way}"

    def llm_format(self, text: str):
        prompt = inspect.cleandoc("""
            请优化这段Markdown的格式，严格按照以下要求优化：
            1. 根据语义合并误断的行，包括标题和段落的误断
            2. 删除多余的空行
            3. 不要修改任何图片元素
            4. 不要输出任何多余内容和说明，仅仅输出优化后的结果
        """)
        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": text},
            ]
        }]
        response = self.llm.invoke(message)
        return response.content


    def get_image_des(self, image_base64: str):
        prompt = inspect.cleandoc("""
                    请分析这张文档图片，要求：
                    1. 描述图片内容
                    2. 如果是流程图，描述流程
                    3. 如果是架构图，描述系统关系
                    4. 使用一段纯文本描述，不要换行
                    5. 简洁准确
                """)

        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "base64": image_base64,
                    "mime_type": "image/png",
                },
            ]
        }]

        response = self.vlm.invoke(message)
        return response.content
