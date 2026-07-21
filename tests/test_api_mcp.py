from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api_mcp import (
    extract_final_answer,
    extract_json_candidate,
    extract_sources_from_tool_messages,
    message_content_to_text,
)

# 測試 extract_json_candidate 解析引用資料功能(作為備用方案)
# 測試包含：正常輸入、markdown文本、無輸入
def test_extract_json_candidate_from_plain_json():
    raw = '[{"content":"abc","score":0.9}]'

    result = extract_json_candidate(raw)

    assert result == raw


def test_extract_json_candidate_from_markdown_code():
    raw = """```json
        [{"content":"abc","score":0.9}]
    ```"""

    result = extract_json_candidate(raw)

    assert result == '[{"content":"abc","score":0.9}]'


def test_extract_json_candidate_from_no_json_exists():
    raw = "this is just plain text"

    result = extract_json_candidate(raw)

    assert result is None

# 測試 message_content_to_text 將模型輸入解析成純文字功能
# 測試包含：輸入str、輸入list[dict]、輸入夾雜其他type
def test_message_content_to_text_returns_same_string_for_plain_text():
    result = message_content_to_text("hello world")

    assert result == "hello world"


def test_message_content_to_text_joins_text_blocks_from_list():
    content = [
        {"type": "text", "text": "第一段"},
        {"type": "text", "text": "第二段"},
    ]

    result = message_content_to_text(content)

    assert result == "第一段\n第二段"


def test_message_content_to_text_ignores_non_text_blocks_without_text_field():
    content = [
        {"type": "text", "text": "第一段"},
        {"type": "image", "url": "https://example.com/a.png"},
        {"type": "text", "text": "第二段"},
    ]

    result = message_content_to_text(content)

    assert result == "第一段\n第二段"

# 測試 extract_final_answer 取出最終回答功能。
def test_extract_final_answer_from_ai_message_content():
    messages = [
        HumanMessage(content="你好"),
        AIMessage(content="第一個回答"),
        HumanMessage(content="再問一次"),
        AIMessage(content="最後回答"),
    ]

    result = extract_final_answer(messages)

    assert result == "最後回答"


def test_extract_final_answer_from_no_ai_message_exists():
    messages = [
        HumanMessage(content="只有使用者訊息"),
    ]

    result = extract_final_answer(messages)

    assert result == ""

# 測試 extract_sources_from_tool_messages 解析 tools 引用資料功能
# 測試包含：從主要管道(ToolMessage)解析引用資料、從備用管道(message_content_to_text)解析資料。
def test_extract_sources_from_tool_messages_reads_structured_content_artifact():
    tool_message = ToolMessage(
        content="tool output",
        tool_call_id="call_1",
        artifact={
            "structured_content": {
                "result": [
                    {
                        "rank": 1,
                        "content": "這是第一筆來源",
                        "score": 0.91,
                    },
                    {
                        "rank": 2,
                        "content": "這是第二筆來源",
                        "score": 0.82,
                    },
                ]
            }
        },
    )

    result = extract_sources_from_tool_messages([tool_message])

    assert len(result) == 2
    assert result[0]["rank"] == 1
    assert result[0]["content"] == "這是第一筆來源"
    assert result[0]["score"] == 0.91


def test_extract_sources_from_tool_messages_reads_sources_from_fallback_json_text():
    tool_message = ToolMessage(
        content='[{"rank":1,"content":"fallback source","score":0.75}]',
        tool_call_id="call_2",
    )

    result = extract_sources_from_tool_messages([tool_message])

    assert len(result) == 1
    assert result[0]["rank"] == 1
    assert result[0]["content"] == "fallback source"
    assert result[0]["score"] == 0.75