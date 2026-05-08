# 文档提取主流程: Docling(Document AST) => VLM(Markdown)
import io
import json
import base64
from pathlib import Path
from typing import List, Dict, Any

import fitz
from PIL import Image
import ollama

from docling.document_converter import DocumentConverter


class DocConvertor:

    def __init__(
            self,
            model_name: str = "qwen3-vl",
            output_dir: str = "./output",
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.converter = DocumentConverter()

    # =========================================================
    # Public
    # =========================================================

    def convert_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        主入口
        """

        pdf_path = Path(pdf_path)

        # 1. Docling解析
        result = self.converter.convert(str(pdf_path))

        # 2. 打开PDF
        pdf_doc = fitz.open(str(pdf_path))

        # 3. markdown结果
        markdown_parts = []

        # 4. 遍历文档元素
        for item in result.document.iterate_items():

            try:
                processed = self._process_item(
                    item=item,
                    pdf_doc=pdf_doc,
                )

                if processed:
                    markdown_parts.append(processed)

            except Exception as e:
                print(f"[ERROR] item process failed: {e}")

        final_markdown = "\n\n".join(markdown_parts)

        return {
            "markdown": final_markdown
        }

    # =========================================================
    # Item Process
    # =========================================================

    def _process_item(
            self,
            item,
            pdf_doc,
    ) -> str:

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
            return self._process_table(item, pdf_doc)

        # 图片 / figure
        elif label in ["picture", "figure", "image"]:
            return self._process_figure(item, pdf_doc)

        return ""

    # =========================================================
    # Text
    # =========================================================

    def _extract_text(self, item) -> str:

        text = getattr(item, "text", "")

        if not text:
            return ""

        return text.strip()

    # =========================================================
    # Table
    # =========================================================

    def _process_table(self, item, pdf_doc) -> str:

        image = self._crop_item_image(item, pdf_doc)

        if image is None:
            return ""

        image_path = self._save_temp_image(image)

        prompt = """
你是专业文档解析助手。

请完成：

1. 提取表格全部内容
2. 恢复正确行列关系
3. 使用Markdown表格输出
4. 不要遗漏表头
5. 不要编造数据
"""

        result = self._vl_inference(
            image_path=image_path,
            prompt=prompt,
        )

        return result

    # =========================================================
    # Figure
    # =========================================================

    def _process_figure(self, item, pdf_doc) -> str:

        image = self._crop_item_image(item, pdf_doc)

        if image is None:
            return ""

        image_path = self._save_temp_image(image)

        prompt = """
请分析这张文档图片。

要求：

1. 描述图片内容
2. 如果是流程图，描述流程
3. 如果是架构图，描述系统关系
4. 使用Markdown输出
5. 简洁准确
"""

        result = self._vl_inference(
            image_path=image_path,
            prompt=prompt,
        )

        return f"\n[FIGURE]\n{result}\n"

    # =========================================================
    # Crop
    # =========================================================

    def _crop_item_image(self, item, pdf_doc):

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
                [pix.width, pix.height],
                pix.samples,
            )

            return image

        except Exception as e:
            print(f"[ERROR] crop failed: {e}")
            return None

    # =========================================================
    # VL Inference
    # =========================================================

    def _vl_inference(
            self,
            image_path: str,
            prompt: str,
    ) -> str:

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path],
                }
            ]
        )

        return response["message"]["content"]

    # =========================================================
    # Utils
    # =========================================================

    def _save_temp_image(self, image: Image.Image) -> str:

        temp_path = self.output_dir / "temp_crop.png"

        image.save(temp_path)

        return str(temp_path)


if __name__ == "__main__":
    converter = DocConvertor(
        model_name="qwen3-vl"
    )

    result = converter.convert_pdf("test.pdf")

    print(result["markdown"])
