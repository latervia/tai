import io
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Literal

import fitz
import pymupdf4llm

from app.shared.logger import logger
from app.infrastructure.llm import ollama_vl, qwen
from app.infrastructure.storage.minio import get_minio_storage
from app.infrastructure.storage.base import BaseStorage
from app.domain.rag.convert.base import BaseConvertor


class FitzConvertor(BaseConvertor):

    def __init__(self):
        super().__init__(way="pymupdf4llm", llm=qwen(), vlm=ollama_vl())
        self.storage: BaseStorage = get_minio_storage()
        self.image_pattern = r"\*\*==> .*? \[\d+ x \d+\] .*? <==\*\*"

    def convert(
            self,
            doc_id: str,
            source_type: Annotated[Literal["pdf", "docx"], "源文件格式"] = "pdf",
            target_type: Annotated[Literal["txt", "html", "md"], "目标文件格式"] = "md"
    ) -> str:
        source_name = self.get_source_name(doc_id, source_type)
        logger.info(f"开始转换文档: {source_name}")

        target_name = self.get_target_name(doc_id, target_type)
        local_path = self.storage.download_local(source_name)
        logger.info(f"文件已下载到: {local_path}")

        try:
            doc_md = pymupdf4llm.to_markdown(
                local_path,
                header=False,
                footer=False,
                dpi=300
            )

            placeholders = self._get_image_placeholders(doc_md)
            image_items = self._process_images(doc_id, local_path, placeholders)

            doc_md = self._replace_placeholders(doc_md, placeholders, image_items)
            doc_md = self.llm_format(doc_md)

            self.storage.upload(io.BytesIO(doc_md.encode("utf-8")), target_name)
            logger.info(f"文档转换完成: {target_name}")
            return doc_md

        finally:
            try:
                os.unlink(local_path)
            except OSError:
                pass

    def _get_image_placeholders(self, text: str) -> list[str]:
        return re.findall(self.image_pattern, text)

    def _process_images(
            self,
            doc_id: str,
            local_path: str,
            placeholders: list[str]
    ) -> list[str]:
        doc = fitz.open(local_path)
        try:
            page_count = doc.page_count

            tasks: list[tuple[int, int, bytes]] = []
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    image_bytes = doc.extract_image(xref)["image"]
                    tasks.append((page_num, img_index, image_bytes))

            if not tasks:
                return []

            results: list[tuple[int, str]] = []
            with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as executor:
                future_to_idx = {}
                for idx, (page_num, img_index, image_bytes) in enumerate(tasks):
                    object_name = f"{self.get_assets_dir(doc_id)}/{page_num + 1}_{img_index + 1}.png"
                    future = executor.submit(self._process_single_image, image_bytes, object_name)
                    future_to_idx[future] = idx

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results.append((idx, future.result()))

            results.sort(key=lambda x: x[0])
            return [item for _, item in results]
        finally:
            doc.close()

    def _process_single_image(self, image_bytes: bytes, object_name: str) -> str:
        self.storage.upload(io.BytesIO(image_bytes), object_name)
        description = self.get_image_des(image_bytes)
        return f"![{description}]({object_name})"

    def _replace_placeholders(
            self,
            doc_md: str,
            placeholders: list[str],
            image_items: list[str]
    ) -> str:
        if len(placeholders) != len(image_items):
            logger.warning(
                f"图片数量不一致: 占位符 {len(placeholders)} 个，实际处理 {len(image_items)} 个"
            )
            count = min(len(placeholders), len(image_items))
            for i in range(count):
                doc_md = doc_md.replace(placeholders[i], image_items[i])
        else:
            for placeholder, item in zip(placeholders, image_items):
                doc_md = doc_md.replace(placeholder, item)
        return doc_md
