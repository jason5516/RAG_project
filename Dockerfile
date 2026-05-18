FROM python:3.11-slim

WORKDIR /app

# 安裝系統依賴（ChromaDB 需要）
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安裝額外依赖（rank_bm25、fastapi、uvicorn）
RUN pip install --no-cache-dir rank-bm25 fastapi uvicorn

# 複製 src 目錄
COPY src/ ./src/

# 複製 chroma_db（如果有的話）
COPY chroma_db/ ./chroma_db/

# 環境變數（生產環境需要手動設定）
ENV MINIMAX_API_KEY=your_api_key_here

# Render 需要的通訊埠
EXPOSE 8000

# 啟動命令
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]