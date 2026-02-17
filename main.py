import io
import os
import urllib.parse

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv

from agent import stream_bizz_bot_response, get_bizz_bot_response
from voice_handler import handle_voice_conversation

load_dotenv()

app = FastAPI()

#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-AI-Text", "X-User-Text"],
)

#DB
client = MongoClient(os.getenv("MONGODB_ATLAS_URI"))
db     = client["bizz_bot_db"]



# MODELS


class ChatRequest(BaseModel):
    user_id: str
    chat_id: str
    message: str



#  TEXT CHAT — SSE STREAMING

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Streams the AI response as Server-Sent Events (SSE).

    SSE event types the frontend receives:
      - "data: <token>\\n\\n"         — a word/token to append to the AI bubble
      - "data: \\n\\n\\n\\n"          — a newline (frontend converts \\n to actual newline)
      - "data: [VOICE] <text>\\n\\n"  — the short voice summary (optional use)
      - "data: [DONE]\\n\\n"          — stream is complete
    """
    return StreamingResponse(
        stream_bizz_bot_response(req.user_id, req.chat_id, req.message),
        media_type="text/event-stream",
        headers={
            # Prevent buffering — critical for SSE to work in real time
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )



#  VOICE CHAT — unchanged, not streaming


@app.post("/voice-chat")
async def voice_chat_endpoint(
    file:    UploadFile = File(...),
    user_id: str        = Form(...),
    chat_id: str        = Form(...),
):
    try:
        audio_data = await file.read()

        wav_bytes, visual_report, user_text = await handle_voice_conversation(
            audio_data, user_id, chat_id
        )

        headers = {
            "X-AI-Text":   urllib.parse.quote(str(visual_report)),
            "X-User-Text": urllib.parse.quote(str(user_text)),
        }

        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers=headers,
        )

    except Exception as e:
        print(f"[/voice-chat] Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})



#  CHAT HISTORY ENDPOINTS


@app.get("/chats/{user_id}")
async def get_user_chats(user_id: str):
    chats = db.users_chats.find({"user_id": user_id}).sort("updated_at", -1)
    return [
        {
            "chat_id":    c["chat_id"],
            "title":      c.get("title", "New Strategic Chat"),
            "updated_at": c["updated_at"].isoformat() if c.get("updated_at") else None,
        }
        for c in chats
    ]


@app.get("/chat/{chat_id}")
async def get_specific_chat(chat_id: str):
    chat = db.users_chats.find_one({"chat_id": chat_id})
    if not chat:
        return {"messages": []}

    messages = []
    for msg in chat.get("messages", []):
        messages.append({
            "role":      msg.get("role", "human"),
            "content":   msg.get("content", ""),
            "timestamp": msg["timestamp"].isoformat() if msg.get("timestamp") else None,
        })
    return {"messages": messages}


@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    result = db.users_chats.delete_one({"chat_id": chat_id})
    if result.deleted_count:
        return {"message": "Chat deleted successfully."}
    raise HTTPException(status_code=404, detail="Chat not found.")