import os
from dotenv import load_dotenv
load_dotenv()

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import src.api as api

import re

api.initialize()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


# 指令：/start
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("你好！我是 RAG 助理，請輸入你的問題。")

# 指令：/help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("傳訊息給我，我會根據文件資料回答。")

# 處理一般訊息
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    user_message = update.message.text
    await update.message.reply_text("處理中...")

    # 取出歷史對話
    history_texts = api.conversation_history.load_memory_variables({}).get("history", "")

    # 意圖分類
    intent = api.classify_intent(api.ChatRequst(query=user_message), history_texts)

    if intent == "document":
        results = api.hybird_search(user_message, k=5, alpha=0.4)
        context_chunks = [text for text, _ in results]
        context = "\n\n".join(context_chunks)
        sources = [chunk[:500].replace("\n", " ") for chunk in context_chunks]
        prompt = f"""根據以下參考資料回答問題。請用繁體中文回答。

            歷史對話：{history_texts}
            
            問題：{user_message}
            
            資料：{context}
            
            回答：
            
        """
    else:
        sources = ["本輪對話無需引用"]
        prompt = f"""檢查歷史對話並回答問題。請用繁體中文回答用戶。

            歷史對話：{history_texts}
            
            問題：{user_message}
            
            回答：
            
        """

    # 呼叫 agent (LLM)
    full_answer = ""
    last_answer = ""
    for chunk in api.agent.stream(
        {'messages': [{'role': 'user', 'content': prompt}]},
        stream_mode='updates'
    ):
        if 'model' in chunk:
            for msg in chunk['model']['messages']:
                full_answer += msg.content
                last_answer = msg.content
                print(f"======原始訊息======\n{full_answer}")
    
    think = re.findall(r'<think>(.*?)</think>', full_answer, re.DOTALL)
    final_answer = re.sub(r'<think>.*?</think>', "", last_answer, flags=re.DOTALL)

    # 將本輪對話加入memory並更新
    api.conversation_history.save_context(
        {"input": user_message},
        {"output": final_answer}
    )

    await update.message.reply_text(final_answer)

#  啟動 Bot
app = Application.builder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start_command))
app.add_handler(CommandHandler("help", help_command))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

print("Telegram Bot 啟動中...")
app.run_polling()