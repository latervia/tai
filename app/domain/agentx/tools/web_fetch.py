"""
网页抓取工具。

当前为桩实现 —— 接入真实抓取（Playwright / Jina Reader / Firecrawl）时替换此函数。
"""


async def web_fetch(url: str, *, max_chars: int = 8000) -> dict:
    """抓取网页正文，返回 title / url / text。

    桩实现：返回占位内容，供管线联调。
    """
    return {
        "title": f"页面标题 — {url}",
        "url": url,
        "text": (
            f"这是 {url} 的模拟正文内容。" * 40
        )[:max_chars],
    }
