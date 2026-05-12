import io
import re
from typing import Annotated, Literal

import fitz
import pymupdf4llm

from app.core.logger import logger
from app.core.storage.minio_storage import get_minio_storage
from app.core.storage.storage import BaseStorage
from app.rag.convert.base import BaseConvertor


class FitzConvertor(BaseConvertor):

    def __init__(self):
        super().__init__("pymupdf4llm")
        self.storage: BaseStorage = get_minio_storage()
        self.image_pattern = r"\*\*==> .*? \[\d+ x \d+\] .*? <==\*\*"

    def convert(
            self,
            doc_id: str,
            source_type: Annotated[Literal["pdf", "docx"], "源文件格式"] = "pdf",
            target_type: Annotated[Literal["txt", "html", "md"], "目标文件格式"] = "md"
    ) -> str:
        source_name = self.get_source_name(doc_id, source_type)

        print(f"下载文件名称: {source_name}")

        target_name = self.get_target_name(doc_id, target_type)

        # file_path = self.storage.get_url(source_name)
        local_path = self.storage.download_local(source_name)
        print(f"文件保存路径: {local_path}")

        doc_md = pymupdf4llm.to_markdown(
            local_path,
            # embed_images=True,  # 嵌入Base64图片
            header=False,
            footer=False,
            dpi=300  # 提高图片清晰度
        )

        # 处理图片
        images = self._get_images(doc_md)  # 占位符

        image_list = []  # OSS真实图片object_name
        doc = fitz.open(local_path)
        page_count = doc.page_count
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            page_image_list = page.get_images(full=True)
            for img_index, img in enumerate(page_image_list):
                xref = img[0]
                base64_image = doc.extract_image(xref)
                image_bytes = base64_image["image"]
                object_name = f"{self.get_assets_dir(doc_id)}/{page_num + 1}_{img_index + 1}.png"
                self.storage.upload(io.BytesIO(image_bytes), object_name)

                # markdown图片: ![替代文字](图片路径)
                # 1.调用大模型获取图片描述文字
                image_des = self.get_image_des(image_bytes)
                image_item = f"![{image_des}]({object_name})"

                image_list.append(image_item)

        if len(images) != len(image_list):
            logger.warning(f"图片数量不一致，源文件: {len(images)}，目标文件: {len(image_list)}")

        # 替换图片
        image_num = len(images)
        for i in range(image_num):
            doc_md = doc_md.replace(images[i], image_list[i])

        # 让大模型处理误断的行
        doc_md = self.llm_format(doc_md)

        self.storage.upload(io.BytesIO(doc_md.encode("utf-8")), target_name)

        return doc_md

    def _get_images(self, text):
        images = re.findall(self.image_pattern, text)
        return images
