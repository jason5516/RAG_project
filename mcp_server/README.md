# 架設 MCP Server 

```mcp_server/``` 在第一階段用於學習建立最小 MCP Server，先不整合既有 RAG 流程。 

## Tools (工具規格)

```python
echo(text: str) -> str : 回傳原始文字檔

add(a: number, b: number) -> number : 回傳兩數相加結果

# 新增既有功能
search_docs(query: str, k: int = 5) -> list[dict] : 進行RAG文件的查找，回傳前 k 筆分數高的資料以及其分數。

calculator(exp: str) -> str : 用於計算數學式。
```
出現錯誤則將訊息寫進stderr。

## 執行方式
```bash
mcp dev mcp_server/main.py
```


## 驗收標準
- 正常啟動Server (以驗證成功)
- client/inspector 能列出 echo、add 兩項功能 (以驗證成功)
- 能成功呼叫兩個工具並得到正確結果 (以驗證成功)
- 新增既有功能 search_docs 並成功呼叫 (以驗證成功)


### 注意事項
- stdio 模式下不要把 log 輸出到 stdout