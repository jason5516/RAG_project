from mcp.server.fastmcp import FastMCP
import sys

# 建立 MCP Server 實例
mcp = FastMCP("rag-project-demo")


@mcp.tool()
def echo(text: str) -> str:
    """回傳原始文字。"""
    return text


@mcp.tool()
def add(a: float, b: float) -> float:
    """回傳兩個數字相加的結果。"""
    return a + b


def main() -> None:
    # stdio 模式下，stdout 會被 MCP 協定使用
    # 如果要除錯，請把訊息寫到 stderr
    print("Starting MCP server...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()