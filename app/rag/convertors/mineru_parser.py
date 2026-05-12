import pathlib

import fitz
import pymupdf
import pymupdf4llm
from langchain_mineru import MinerULoader

def test1():
    loader = MinerULoader(
        # source="C:/_src_/tai/doc/Qwen-Scope.pdf",
        source="C:/_src_/tai/doc/人事核心FAQ.pdf",
        mode="precision",
        token="eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI2MjYwMDgwNSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3ODU1NTc4NywiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiNjkxMmM2ZDUtM2ViZi00NWNlLThmZTAtNzQ0NmI3MjJjZGZmIiwiZW1haWwiOiIiLCJleHAiOjE3ODYzMzE3ODd9.PNFGbGIfX6-VQHdWaKQLcxOoyiFDpisTS6e9uDUal8xb3CRcfFPglTrB3riH1HoAbFc9dHeCfHcKgn_gm4BCsw",
        # or set MINERU_TOKEN
        # language="ch",
        # split_pages=True,
        # pages="1-5",
        # timeout=300,
    )

    docs = loader.load()
    for doc in docs:
        print("-" * 100)
        print(f"{doc.page_content[:200]}")


def test2():
    input_pdf = "C:/_src_/tai/doc/人事核心FAQ.pdf"
    # 定义图片存放的目录
    image_dir = "images"

    # 使用 to_markdown 并传入图片相关参数
    doc_md = pymupdf4llm.to_markdown(
        input_pdf,
        embed_images=True,
        # write_images=True,  # 关键：开启图片提取
        # image_path=image_dir,  # 图片保存的子目录名
        # image_format="png",  # 图片格式
        header=False,
        footer=False,
        dpi=300  # 可选：提高图片清晰度（默认较小）
    )
    #
    # print(doc_md)
    # 如果想保存成文件
    pathlib.Path("output.md").write_text(doc_md, encoding="utf-8")

    doc = fitz.open(input_pdf)
    image_map = {}
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        # images = page.get_images(full=True)
        #
        # for image in images:
        #     xref = image[0]
        #     base_image = doc.extract_image(xref)
        #     pathlib.Path(f"{image_dir}/{xref}.png").write_bytes(base_image["image"])
        #     print(1)

        image_list = page.get_images(full=True)


        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            # 构造对象名：例如 pdf_name/page1_img0.png
            obj_name = f"hr_faq/page{page_num}_img{img_index}.{ext}"

            # 上传并记录链接
            # minio_url = upload_to_minio(image_bytes, obj_name)

            # 这里的 Key 需要匹配 pymupdf4llm 生成的默认占位符规则
            # 提示：pymupdf4llm 默认生成的图片占位符通常是 images/xxx
            # 我们可以通过正则匹配来精准替换
            local_placeholder = f"images/page{page_num + 1}-{img_index}.{ext}"
            image_map[local_placeholder] = "minio_url" + "/" + obj_name

        print(123)

    for local_path, remote_url in image_map.items():
        md_text = doc_md.replace(local_path, remote_url)

    print(md_text)
    print(12)



def get_toc():
    doc = fitz.open("C:/_src_/tai/doc/人事核心FAQ.pdf")
    toc = doc.get_toc()  # 获取目录
    for level, title, page in toc:
        print(f"层级: {level}, 标题: {title}, 对应页码: {page}")


if __name__ == '__main__':
    test2()
