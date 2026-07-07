import sys
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP
from src.hybard_search import hybird_search

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

@mcp.tool()
def search_docs(query: str, k: int) -> list[dict]:
    """進行RAG文件的查找，回傳前 k 筆分數高的資料以及其分數。"""
    ref = hybird_search(query, k)
    result = [
        {
            "content": text,
            "score": float(score),
        }
        for text, score in ref
    ]
    

    return result

@mcp.tool()
def calculator(exp: str) -> str:
    """執行數學計算。expression 是數學式，例如 '2+2'。"""
    try:
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', exp):
            return "計算錯誤：只支援數學運算"
        result = eval(exp)
        return str(result)
    except ZeroDivisionError:
        return "計算錯誤：除數不能為零"
    except:
        return "計算錯誤：無效的表達式"



def main() -> None:
    # stdio 模式下，stdout 會被 MCP 協定使用
    # 如果要除錯，請把訊息寫到 stderr
    print("Starting MCP server...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
