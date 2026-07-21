import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mcp_server.main as mcp_main

# 測試 calculator 功能
# 測試包含：合法算式、非法算式、除以零錯誤
def test_calculator_for_valid_expression():
    result = mcp_main.calculator("(2 + 3) * 4")

    assert result == '20'

def test_calculator_for_invalid_expression():
    result = mcp_main.calculator("這是測試運算")

    assert result == "計算錯誤：只支援數學運算"

def test_calculator_for_zero_division():
    result = mcp_main.calculator("10 / 0")

    assert result == "計算錯誤：除數不能為零"

# 測試 search_docs 搜尋文件功能。
# 測試包含：讀取預設文件格式、無內容返回
def test_search_docs_from_structured_results(monkeypatch):
    fake_results = [
        ("第一段文件內容", 0.91),
        ("第二段文件內容", 0.82),
    ]

    def fake_hybird_search(query, k=5):
        assert query == "學貸條件"
        assert k == 2
        return fake_results

    monkeypatch.setattr(mcp_main, "hybird_search", fake_hybird_search)

    result = mcp_main.search_docs("學貸條件", 2)

    assert isinstance(result, list)
    assert len(result) == 2

    assert result[0]["rank"] == 1
    assert result[0]["content"] == "第一段文件內容"
    assert result[0]["score"] == 0.91

    assert result[1]["rank"] == 2
    assert result[1]["content"] == "第二段文件內容"
    assert result[1]["score"] == 0.82

def test_search_docs_from_no_results(monkeypatch):
    def fake_hybird_search(query, k=5):
        return []

    monkeypatch.setattr(mcp_main, "hybird_search", fake_hybird_search)

    result = mcp_main.search_docs("不存在的查詢", 5)

    assert result == []