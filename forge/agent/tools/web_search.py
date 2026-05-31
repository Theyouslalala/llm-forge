from ..base_agent import BaseTool


class WebSearchTool(BaseTool):
    """Simulated web search tool for demonstration."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "搜索互联网获取信息。输入搜索关键词，返回相关搜索结果摘要。"

    def run(self, input_text: str) -> str:
        query = input_text.strip()
        results = [
            {"title": f"搜索结果1: {query}", "snippet": f"关于'{query}'的相关信息..."},
            {"title": f"搜索结果2: {query}", "snippet": f"'{query}'的详细解释..."},
        ]
        output = f"搜索查询: {query}\n\n"
        for i, r in enumerate(results, 1):
            output += f"[{i}] {r['title']}\n    {r['snippet']}\n\n"
        return output
