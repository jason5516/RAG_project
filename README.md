# RAG Chat API
這是一個簡易的RAG專案，將data/下的檔案經過chunk與embedding轉成知識向量庫儲存起來，接著通過相似度查找與BM25的混合指標找出
最接近的k個chunks作為儲備知識加入至prompt中，讓模型能根據data/中的檔案回答問題。

## 技術架構
- FastAPI
- ChromaDB
- 使用Minimax-M2.7作為基礎語言模型
- Hybrid Search（BM25 + BGE embedding）

## 如何啟動?
首先通過conda建立環境，並通過requirements.txt安裝環境：`pip install -r requirements.txt`
使用`uvicorn src.api:app --reload`啟動服務，在初始化時會自動建立資料庫並載入模型。

通過下列指令傳入問題。
```
curl -X POST http://localhost:8000/chat \
      -H "Content-Type: application/json" \
      -d '{"query": "你的問題"}'
```



