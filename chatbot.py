import sqlite3
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# 1. THE MODEL
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.7)


# 2. HELPER: turn model output into plain text
def get_text(content) -> str:
    """Return plain text whether content is a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


# 3. THE STATE
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 4. THE NODE
def chat_node(state: ChatState):
    reply = llm.invoke(state["messages"])
    return {"messages": [reply]}


# 5. THE MEMORY: a real file now (permanent)
connection = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)

# 6. THE GRAPH
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# 7. NEW: ask the database "which chats exist?"
def get_all_threads():
    """Return all chat names saved in the database (oldest first)."""
    found = []
    for saved in checkpointer.list(None):          # None = all chats
        tid = saved.config["configurable"]["thread_id"]
        if tid not in found:                        # skip repeats
            found.append(tid)
    found.reverse()                                 # list() gives newest first
    return found