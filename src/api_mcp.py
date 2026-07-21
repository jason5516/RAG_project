import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationSummaryMemory
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
import re

import json
from typing import Any
from langchain_core.messages import AIMessage, ToolMessage

# 全域變數
agent = None
conversation_history = None
tavily = None
llm = None

# 用於 fallback 方案，從 raw_data 中取出 json 格式檔案
def extract_json_candidate(raw_text: str) -> str | None:
    raw_text = raw_text.strip()

    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

    # 優先找 list JSON
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return raw_text[start:end + 1]

    # 其次找 dict JSON
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_text[start:end + 1]

    return None

# 將模型輸出轉成文字，方便之後做思考過程與最終回答的解析
def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(part for part in parts if part)

    return str(content)

# 取得模型最終回答
def extract_final_answer(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message_content_to_text(message.content).strip()
    return ""

# 假設模型有使用 doc_search() 將引用資料回傳
def extract_sources_from_tool_messages(messages: list[Any]) -> list[dict]:
    sources = []

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        artifact = getattr(message, "artifact", None)
        structured = None
        if isinstance(artifact, dict):
            structured = artifact.get("structured_content")

            for item in structured:
                if isinstance(structured[item], list):
                    for resource in structured[item]:
                        if 'content' in resource:
                            sources.append({
                                "content": str(resource["content"][:500]).strip(),
                                "rank": resource["rank"],
                                "score": resource["score"],
                            })
            continue
        
        # Fallback 如果沒有 structured_content 或是格式不同，從原始訊息作為 source。 
        raw_text = message_content_to_text(message.content).strip()
        json_candidate = extract_json_candidate(raw_text)
        try:
            parsed = json.loads(json_candidate)

            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "content" in item:
                        content = str(item["content"]).strip()
                        sources.append({
                            "content": content[:500].replace("\n", " "),
                            "rank": len(sources) + 1,
                            "score": item["score"],
                            "source_type": "fallback"
                        })

            elif isinstance(parsed, dict):
                result_items = parsed.get("result")
                if isinstance(result_items, list):
                    for item in result_items:
                        if isinstance(item, dict) and "content" in item:
                            content = str(item["content"]).strip()
                            if content:
                                sources.append(content[:500].replace("\n", " "))

        except json.JSONDecodeError:
            pass

    return sources


async def initialize():
    # 使用全域變數
    global embedding, db, bm25, texts, agent, conversation_history, tavily, llm

    # 載入llm模型
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    llm = ChatOpenAI(
        model="MiniMax-M2.7",
        openai_api_key=api_key,
        openai_api_base="https://api.minimax.io/v1",
        temperature=0.7
    )

    conversation_history = ConversationSummaryMemory(llm=llm)

    client = MultiServerMCPClient(
        {
            "rag_tools": {
                "transport": "stdio",
                "command": "/home/jason/miniconda3/envs/airag/bin/python",
                "args": [
                    "/home/jason/.openclaw/workspace/projects/llm_jason5516_projects/RAG_project/mcp_server/main.py"
                ],
            }
        }
    )

    mcp_tools = await client.get_tools()
    agent = create_agent(
        model=llm,
        tools=mcp_tools,
        system_prompt="你是個問答助手",
    )

    print("✅ MiniMax LLM 初始化完成")



@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時進行初始化
    await initialize()
    yield
    

app = FastAPI(title="RAG chat API", lifespan=lifespan)

class ChatRequst(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    think: list[str] = []
    sources: list[dict] = []

@app.get("/health")
async def health():
    return {"status":"ok"}


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequst):
    
    # 取出歷史對話
    history_texts = conversation_history.load_memory_variables({}).get("history", "")
   
    prompt = f"""
        優先使用 search_docs 工具查找文件內容回答用戶問題。請用繁體中文回答用戶。
        請根據查得內容回答，不要自行捏造。
        歷史對話：{history_texts}
        問題：{req.query}
        回答：
    """

    # 呼叫 agent (LLM)
    full_answer = ""
   
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    
    messages = result["messages"]
    full_answer = extract_final_answer(messages)

    think = re.findall(r"<think>(.*?)</think>", full_answer, re.DOTALL)
    final_answer = re.sub(r"<think>.*?</think>", "", full_answer, flags=re.DOTALL).strip()

    sources = extract_sources_from_tool_messages(messages)
    if not sources:
        sources = ["本輪對話無需引用"]

    # 將本輪對話加入memory並更新
    conversation_history.save_context(
        {"input": req.query},
        {"output": final_answer}
    )

    return ChatResponse(think=think, answer=final_answer, sources=sources)

if __name__ == '__main__':
    import asyncio
    asyncio.run(initialize())
    
