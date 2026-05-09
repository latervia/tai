import base64
import inspect
import io
from typing import Dict, Any

import fitz
from PIL import Image
from docling.document_converter import DocumentConverter

from app.core.llm import ollama
from app.core.minio_manager import MinioManager


class RagDocLayout:
    FILE_ORIGINAL = "original.pdf"  # 原始文件
    FILE_CONTENT = "content.md"  # 解析后的Markdown
    FILE_METADATA = "metadata.json"  # 元数据

    DIR_ASSETS = "assets"  # 存放图片


"""
文档提取主流程: Docling(Document AST) => VLM(Markdown)

bucket-name (如: rag-data)
└── documents/
    └── {doc_id}/
        ├── original.pdf          # 原始文件备份（可选）
        ├── content.md             # 解析后的 Markdown
        ├── metadata.json          # 关键元数据（页数、作者、时间等）
        └── assets/                # 存放该 PDF 提取出的所有图片
            ├── img_p1_1.png       # 命名建议：页码_序号
            ├── img_p5_1.jpg
            └── ...
"""


class DocConvertor:

    def __init__(self, minio: MinioManager):
        self.converter = DocumentConverter()
        self.minio = minio

    def convert_pdf(self, doc_id: str) -> Dict[str, Any]:
        """将PDF文档转换为Markdown格式"""

        original_name = doc_id + "/" + RagDocLayout.FILE_ORIGINAL

        temp_path = self.minio.download_to_temp(original_name)

        # 1.Docling解析
        result = self.converter.convert(temp_path)

        # 2.打开PDF
        pdf_doc = fitz.open(temp_path)

        # 3. markdown结果
        markdown_parts = []

        # 4. 遍历文档元素
        for item in result.document.iterate_items():

            try:
                processed = self._process_item(
                    doc_id=doc_id,
                    item=item,
                    pdf_doc=pdf_doc,
                )

                if processed:
                    markdown_parts.append(processed)

            except Exception as e:
                print(f"[ERROR] item process failed: {e}")

        final_markdown = "\n\n".join(markdown_parts)

        # 存储markdown 观测解析结果 + 减少重复解析
        self.minio.upload(io.BytesIO(final_markdown.encode("utf-8")), f"{doc_id}/{RagDocLayout.FILE_CONTENT}")

    def _process_item(self, doc_id, item, pdf_doc) -> str:
        """处理Docling解析输出的文档元素"""

        label = getattr(item, "label", "")

        # 普通文本
        if label in [
            "text",
            "paragraph",
            "title",
            "section_header",
            "list_item",
            "code",
        ]:
            return self._extract_text(item)

        # 表格
        elif label == "table":
            return self._process_table(doc_id, item, pdf_doc)

        # 图片 / figure
        elif label in ["picture", "figure", "image"]:
            return self._process_figure(doc_id, item, pdf_doc)

        return ""

    def _extract_text(self, item) -> str:
        """从文档元素中提取文本内容"""

        text = getattr(item, "text", "")

        if not text:
            return ""

        return text.strip()

    def _process_table(self, doc_id, item, pdf_doc) -> str:
        """处理文档中的表格元素"""

        image_base64 = self._crop_item_image(doc_id, item, pdf_doc)

        if image_base64 is None:
            return ""

        prompt = inspect.cleandoc("""
            你是专业文档解析助手。

            请完成：
            
            1. 提取表格全部内容
            2. 恢复正确行列关系
            3. 使用Markdown表格输出
            4. 不要遗漏表头
            5. 不要编造数据
        """)

        result = self._vl_inference(image_base64, prompt=prompt)

        return result

    def _process_figure(self, doc_id, item, pdf_doc) -> str:
        """处理文档中的图片元素"""

        image_base64 = self._crop_item_image(doc_id, item, pdf_doc)

        if image_base64 is None:
            return ""

        prompt = inspect.cleandoc("""
            请分析这张文档图片。
            
            要求：
            
            1. 描述图片内容
            2. 如果是流程图，描述流程
            3. 如果是架构图，描述系统关系
            4. 使用Markdown输出
            5. 简洁准确
        """)

        result = self._vl_inference(image_base64, prompt=prompt)

        return f"\n[FIGURE]\n{result}\n"

    def _crop_item_image(self, doc_id, item, pdf_doc):
        """
            1. 根据item的prov信息 fitz从pdf_doc中裁剪出item的图片
            2. 将图片存储到minio
            3. 返回图片Base64编码
        """

        prov = getattr(item, "prov", None)

        if not prov:
            return None

        try:
            page_no = prov[0].page_no - 1

            bbox = prov[0].bbox

            page = pdf_doc[page_no]

            rect = fitz.Rect(
                bbox.l,
                bbox.t,
                bbox.r,
                bbox.b,
            )

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                clip=rect,
            )

            image = Image.frombytes(
                "RGB",
                (pix.width, pix.height),
                pix.samples,
            )

            buffer = io.BytesIO()
            image.save(buffer, format="png")
            buffer.seek(0)

            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # 存储到Minio
            image_name = f"{doc_id}/{RagDocLayout.DIR_ASSETS}/img_p{page_no + 1}_{item.id}.png"
            self.minio.upload(buffer, image_name)

            return image_base64

        except Exception as e:
            print(f"[ERROR] crop failed: {e}")
            return None

    def _vl_inference(self, image_base64: str, prompt: str) -> str:
        """使用 VLM 进行视觉语言推理 获得图片描述"""

        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "base64": image_base64,
                    "mime_type": "image/jpeg",
                },
            ]
        }

        response = ollama().invoke(message)

        return response["message"]["content"]
