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

ai-knowledge-base/doc-id/original.pdf
ai-knowledge-base/doc-id/content.md
ai-knowledge-base/doc-id/metadata.json
ai-knowledge-base/doc-id/assets/img_p1_1.png

"""
import inspect
from typing import Dict, Any

import fitz
import ollama
from PIL import Image
from docling.document_converter import DocumentConverter

from app.core.minio_manager import MinioManager


class RagDocLayout:
    FILE_ORIGINAL = "original.pdf"  # 原始文件
    FILE_CONTENT = "content.md"  # 解析后的Markdown
    FILE_METADATA = "metadata.json"  # 元数据

    DIR_ASSETS = "assets"  # 存放图片


class DocConvertor:

    def __init__(self, minio: MinioManager):
        self.converter = DocumentConverter()
        self.minio = minio

    def convert_pdf(self, doc_id: str) -> Dict[str, Any]:

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

        # todo 存储markdown 观测解析结果 减少重复解析

        return {
            "markdown": final_markdown
        }

    # =========================================================
    # Item Process
    # =========================================================

    def _process_item(self, doc_id, item, pdf_doc) -> str:

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

        text = getattr(item, "text", "")

        if not text:
            return ""

        return text.strip()

    def _process_table(self, doc_id, item, pdf_doc) -> str:

        image = self._crop_item_image(doc_id, item, pdf_doc)

        if image is None:
            return ""

        image_path = self._save_temp_image(image)

        prompt = inspect.cleandoc("""
            你是专业文档解析助手。

            请完成：
            
            1. 提取表格全部内容
            2. 恢复正确行列关系
            3. 使用Markdown表格输出
            4. 不要遗漏表头
            5. 不要编造数据
        """)

        result = self._vl_inference(
            image_path=image_path,
            prompt=prompt,
        )

        return result

    def _process_figure(self, doc_id, item, pdf_doc) -> str:

        image = self._crop_item_image(doc_id, item, pdf_doc)

        if image is None:
            return ""

        image_path = self._save_temp_image(image)

        prompt = inspect.cleandoc("""
            请分析这张文档图片。
            
            要求：
            
            1. 描述图片内容
            2. 如果是流程图，描述流程
            3. 如果是架构图，描述系统关系
            4. 使用Markdown输出
            5. 简洁准确
        """)

        result = self._vl_inference(
            image_path=image_path,
            prompt=prompt,
        )

        return f"\n[FIGURE]\n{result}\n"

    def _crop_item_image(self, doc_id, item, pdf_doc):
        """
        根据item的prov信息，从pdf_doc中裁剪出item的图片
        :param item:
        :param pdf_doc:
        :return:
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

            # TODO 存储到Minio
            image_name = f"{doc_id}/assets/img_p{page_no + 1}_{item.id}.png"

            return image

        except Exception as e:
            print(f"[ERROR] crop failed: {e}")
            return None

    def _vl_inference(
            self,
            image_path: str,
            prompt: str,
    ) -> str:
        """
        使用 VLM 进行视觉语言推理
        :param image_path: 图片路径
        :param prompt: 提示语
        :return: 推理结果
        """

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

    def _save_temp_image(self, image: Image.Image) -> str:

        temp_path = self.output_dir / "temp_crop.png"

        image.save(temp_path)

        return str(temp_path)
