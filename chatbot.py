import sqlite3
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition      # NEW

from tools import ALL_TOOLS                                    # NEW

load_dotenv()

# 1. MODEL
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, max_retries=5)
llm_with_tools = llm.bind_tools(ALL_TOOLS)                     # NEW: tell the AI about its tools


# 2. HELPER
def get_text(content) -> str:
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


# 3. STATE (unchanged)
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 4. NODES
def chat_node(state: ChatState):
    reply = llm_with_tools.invoke(state["messages"])           # CHANGED: use llm_with_tools
    return {"messages": [reply]}

tool_node = ToolNode(ALL_TOOLS)                                # NEW: the tool runner


# 5. MEMORY (unchanged)
connection = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)


# 6. GRAPH
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)                             # NEW

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)      # NEW: tools or END?
graph.add_edge("tools", "chat_node")                           # NEW: result goes back to the AI

chatbot = graph.compile(checkpointer=checkpointer)


# 7. List saved chats (unchanged)
def get_all_threads():
    found = []
    for saved in checkpointer.list(None):
        tid = saved.config["configurable"]["thread_id"]
        if tid not in found:
            found.append(tid)
    found.reverse()
    return found