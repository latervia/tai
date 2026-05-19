"""
Research 专用的工具函数。

当前为桩实现 —— 接入真实搜索 API 时替换对应函数即可。
"""


async def web_search(query: str, *, max_results: int = 5) -> list[dict]:
    """搜索网页，返回 title / url / snippet 列表。

    桩实现：返回占位结果，供管线联调。
    接入 Tavily / SerpAPI / Brave Search 时替换此函数。
    """
    return [
        {
            "title": f"关于「{query}」的搜索结果 {i + 1}",
            "url": f"https://example.com/result-{i + 1}?q={query}",
            "snippet": f"这是关于「{query}」的第 {i + 1} 条模拟摘要。接入真实搜索 API 后将被替换。",
        }
        for i in range(min(max_results, 5))
    ]
