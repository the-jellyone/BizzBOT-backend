import os
import asyncio
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import AsyncGenerator

from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

#DATABASE SETUP 
uri= os.getenv("MONGODB_ATLAS_URI")
client= MongoClient(uri)
db = client["bizz_bot_db"]


#LLM CONFIG
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=512,
)


#SERPAPI SEARCH TOOL
def serpapi_search(query: str) -> str:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY is not set."

    params = {"q": query, "api_key": api_key, "engine": "google", "num": 5}
    headers = {"Accept-Encoding": "identity"}

    try:
        response = requests.get(
            "https://serpapi.com/search", params=params, headers=headers, timeout=15
        )
        response.raise_for_status()
        data = response.json()

        results = []
        if "answer_box" in data:
            box = data["answer_box"]
            answer = box.get("answer") or box.get("snippet") or box.get("result")
            if answer:
                results.append(f"Direct Answer: {answer}")

        for r in data.get("organic_results", [])[:4]:
            title   = r.get("title", "")
            snippet = r.get("snippet", "")
            link    = r.get("link", "")
            if snippet:
                results.append(f"- {title}: {snippet} ({link})")

        return "\n".join(results) if results else "No results found."

    except requests.exceptions.Timeout:
        return "Search timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Search request failed: {str(e)}"
    except Exception as e:
        return f"Unexpected search error: {str(e)}"


search_tool = Tool(
    name="Search",
    func=serpapi_search,
    description=(
        "Use this to search the web for real-time information: market trends, "
        "fuel prices, news, competitor data, or any fact you don't know. "
        "Input should be a clear, concise search query string."
    ),
)

tools = [search_tool]


#SYSTEM PROMPT
system_message = """You are a business strategy assistant.
Help users with ideas, problem-solving, positioning, growth, operations, and decision-making based on their specific business context.
Be conversational but professional. Keep responses concise, structured, and practical. No small talk. No motivational fluff. No unrelated discussion.
If information is missing, ask one precise clarifying question before answering. Provide actionable steps or clear trade-offs in every answer. Challenge weak assumptions respectfully. Do not exaggerate results or fabricate data.
Focus on outcomes and execution.

SCOPE:
- Only discuss business topics. If asked about anything else, say: "That's outside my scope. What's the business challenge you're working on?"

RESPONSE FORMAT — match the format to the question:

MODE 1 — CONVERSATIONAL (default):
  Use for: greetings, quick opinions, clarifications, simple follow-ups.
  Format: 1-3 sentences. No headers. No lists. Direct and natural.

MODE 2 — STRUCTURED BRIEF:
  Use ONLY when the question needs a real plan or multi-part answer.
  Format:
    Executive Summary: [2-3 sentence big picture]
    Actionable Steps:
    1. ...
    2. ...
  Triggers: "how do I scale", "give me a strategy", "help me plan", "break this down"

Default to MODE 1. Only use MODE 2 when a plan is genuinely needed.

You have access to the following tools:
{tools}

OUTPUT FORMAT — follow exactly:

If you need to use a tool:
Thought: Do I need to use a tool? Yes
Action: Search
Action Input: your search query
Observation: (result appears here)
Thought: Do I need to use a tool? No
Final Answer: your full response here

If you already know the answer:
Thought: Do I need to use a tool? No
Final Answer: your full response here

CRITICAL RULES:
- Always start with "Thought:" — never jump straight to the answer.
- ALL of your response content — bullets, lists, explanations, everything — must go INSIDE "Final Answer:". Never write content before it.
- "Final Answer:" is not a closing sentence. It is where your entire response begins.
- Never write bullet points or body text before "Final Answer:". If you do, it will be lost.
- Always end with "Final Answer:" containing the complete response.
- Valid tools: [{tool_names}]
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}\n\n{agent_scratchpad}"),
])


#AGENT INIT(imp)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=(
        "Your last response was not in the correct format. "
        "You MUST respond with either:\n"
        "Thought: Do I need to use a tool? No\nFinal Answer: <your answer>\n\n"
        "OR if using a tool:\n"
        "Thought: Do I need to use a tool? Yes\nAction: Search\nAction Input: <query>"
    ),
    max_iterations=7,
    max_execution_time=60,
)


#PROCESS OUTPUT
def _process_output(result: dict) -> dict:
    """
    AgentExecutor only puts text after "Final Answer:" into result["output"].
    Bullet points written BEFORE "Final Answer:" in the LLM log are in
    intermediate_steps[-1][0].log — we pull from there for visual.

    visual → full body content (bullets etc) from the LLM log
    voice  → result["output"] i.e. the Final Answer sentence, spoken aloud
    """
    final_answer = result.get("output", "").strip()
    steps = result.get("intermediate_steps", [])

    # Try to get the full LLM log from the last step
    body_text = ""
    if steps:
        last_log = steps[-1][0].log.strip() if steps[-1][0].log else ""
        if last_log:
            # Strip ReAct control lines, keep only the body content
            skip_prefixes = ("Thought:", "Action:", "Action Input:", "Observation:", "Final Answer:")
            lines = last_log.splitlines()
            body_lines = [l for l in lines if not any(l.strip().startswith(p) for p in skip_prefixes)]
            body_text = "\n".join(body_lines).strip()

    # visual = body content if we got it, else fall back to final_answer
    visual_text = body_text if body_text else final_answer

    # voice = the Final Answer sentence — short, natural, speakable
    voice_text = final_answer
    if len(voice_text) > 400:
        voice_text = voice_text[:400].rsplit(".", 1)[0] + "."

    return {"visual": visual_text, "voice": voice_text}


#DB HELPERS
def _load_history(user_id: str, chat_id: str) -> list:
    chat_folder = db.users_chats.find_one({"user_id": user_id, "chat_id": chat_id})
    formatted_history = []
    if chat_folder and "messages" in chat_folder:
        for msg in chat_folder["messages"][-10:]:
            role    = "human" if msg["role"] == "human" else "ai"
            content = msg["content"]
            formatted_history.append((role, content))
    return formatted_history


def _save_human_message(user_id: str, chat_id: str, user_message: str):
    db.users_chats.update_one(
        {"user_id": user_id, "chat_id": chat_id},
        {
            "$push": {
                "messages": {
                    "role":      "human",
                    "content":   user_message,
                    "timestamp": datetime.now(),
                }
            },
            "$set":         {"updated_at": datetime.now()},
            "$setOnInsert": {
                "title":      user_message[:50],
                "user_id":    user_id,
                "chat_id":    chat_id,
                "created_at": datetime.now(),
            },
        },
        upsert=True,
    )


def _save_ai_message(user_id: str, chat_id: str, visual_text: str):
    db.users_chats.update_one(
        {"user_id": user_id, "chat_id": chat_id},
        {
            "$push": {
                "messages": {
                    "role":      "ai",
                    "content":   visual_text,
                    "timestamp": datetime.now(),
                }
            },
            "$set": {"updated_at": datetime.now()},
        },
    )


# STREAMING ENTRY POINT — used by POST /chat
async def stream_bizz_bot_response(
    user_id: str,
    chat_id: str,
    user_message: str,
) -> AsyncGenerator[str, None]:
    try:
        formatted_history = _load_history(user_id, chat_id)
        _save_human_message(user_id, chat_id, user_message)

        result = await agent_executor.ainvoke({
            "input":        user_message,
            "chat_history": formatted_history,
        })

        processed   = _process_output(result)
        visual_text = processed["visual"]
        voice_text  = processed["voice"]

        _save_ai_message(user_id, chat_id, visual_text)

        # Stream word by word
        words = visual_text.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            if not token:
                continue
            yield f"data: {json.dumps(token)}\n\n"
            await asyncio.sleep(0.05)

        safe_voice = voice_text.replace("\n", " ")
        yield f"data: [VOICE] {safe_voice}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        print(f"[agent] STREAMING ERROR: {e}")
        yield f"data: {json.dumps('I hit a snag, Boss. Please try again.')}\n\n"
        yield "data: [DONE]\n\n"


#NON-STREAMING ENTRY POINT — used by voice_handler.py
async def get_bizz_bot_response(user_id: str, chat_id: str, user_message: str) -> dict:
    """Voice pipeline only — returns full dict, no streaming."""
    try:
        formatted_history = _load_history(user_id, chat_id)
        _save_human_message(user_id, chat_id, user_message)

        result = await agent_executor.ainvoke({
            "input":        user_message,
            "chat_history": formatted_history,
        })

        processed = _process_output(result)
        _save_ai_message(user_id, chat_id, processed["visual"])
        return processed

    except Exception as e:
        print(f"[agent] CRITICAL ERROR: {e}")
        return {
            "visual": "I hit a snag, Boss. Please try again.",
            "voice":  "I ran into a problem. Please try again.",
        }