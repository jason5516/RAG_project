# 架設 MCP Server 

```mcp_server/``` 在第一階段用於學習建立最小 MCP Server，先不整合既有 RAG 流程。 

## Tools (工具規格)

```python
echo(text: str) -> str : 回傳原始文字檔

add(a: number, b: number) -> number : 回傳兩數相加結果
```
出現錯誤則將訊息寫進stderr。

## 執行方式


## 驗收標準
- 正常啟動Server (以驗證成功)
- client/inspector 能列出 echo、add 兩項功能 (以驗證成功)
- 能成功呼叫兩個工具並得到正確結果 (以驗證成功)


### 注意事項
- stdio 模式下不要把 log 輸出到 stdout