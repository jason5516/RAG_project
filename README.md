# RAG Chat API with MCP
這是一個以文件問答為核心的 RAG 專案。系統會從本地知識庫中檢索相關內容，並透過 LangChain agent 與 MCP Server 工具完成回答。

## 技術架構
- FastAPI 提供對話 API 與前端頁面
- ChromaDB 作為向量資料庫
- Hybrid Search 結合 BM25 與 embedding 檢索
- MCP Server 封裝 `search_docs`、`calculator` 等工具
- GitHub Actions 自動執行測試
- AI Agent 通過 MCP client 使用工具

## 系統架構
1. 使用者從前端輸入問題
2. FastAPI `/chat` 接收請求
3. `src/api_mcp.py` 建立 LangChain agent
4. agent 透過 MCP client 連接 `mcp_server/main.py`
5. MCP Server 執行 `search_docs` 等工具
6. 工具回傳結構化的引用結果（包含 `rank`、`content`、`score`）
7. API 整理最終回答與引用來源後回傳前端

## 啟動方式

使用 conda 建立環境：
```bash
conda create -n airag python=3.11 -y
conda activate airag
pip install -r requirements.txt
```

## 環境變數
先建立 `.env` 並設定以下變數：
- `MINIMAX_API_KEY` 或是其他模型的 API KEY
- `OPENWEATHER_API_KEY` (天氣查詢功能，若有使用)

啟動API服務：

```bash
uvicorn src.api_mcp:app --reload
```

備註：
開發或測試 MCP Server
```bash
mcp dev mcp_server/main.py
```

## 功能測試
執行所有測試：

```bash
python -m pytest
```

或是分別測試檔案
```bash
python -m pytest tests/test_api_mcp.py -q
python -m pytest tests/test_mcp_server.py -q
python -m pytest tests/test_api_routes.py -q
```

## 後續規劃

- 增加更多 MCP tools
- 補充更完整的 API 測試
- 支援更完整的來源 metadata
- 強化部署與容器化流程