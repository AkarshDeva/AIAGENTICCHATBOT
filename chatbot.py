import sqlite3
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

from tools import ALL_TOOLS

load_dotenv()

# 1. MODEL (with its toolbox)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, max_retries=5)
llm_with_tools = llm.bind_tools(ALL_TOOLS)


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


# 4. THE SYSTEM PROMPT (the bot's rules)
SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful chatbot with tools.\n"
    "- For questions about an uploaded PDF or document, ALWAYS use search_documents first, "
    "and answer only from what it returns.\n"
    "- For latest news or current events, use web_search.\n"
    "- For maths, use calculator.\n"
    "- For weather, use get_weather.\n"
    "- To buy stock, use purchase_stock. The user will be asked to approve it.\n"
    "- For general questions, answer directly without tools.\n"
    "- If the user asks about a document but none is uploaded, ask them to upload a PDF.\n"
    "- Never invent facts about a document. After using a tool, give a clear final answer."
))


# 5. THE NODES
def chat_node(state: ChatState):
    reply = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
    return {"messages": [reply]}

tool_node = ToolNode(ALL_TOOLS)


# 6. THE MEMORY (permanent file)
connection = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)


# 7. THE GRAPH
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# 8. LIST SAVED CHATS
def get_all_threads():
    """Return all chat names saved in the database (oldest first)."""
    found = []
    for saved in checkpointer.list(None):
        tid = saved.config["configurable"]["thread_id"]
        if tid not in found:
            found.append(tid)
    found.reverse()
    return found


# 9. CHECK IF A CHAT IS PAUSED WAITING FOR APPROVAL
def get_pending_approval(thread_id):
    """If this chat is paused waiting for approval, return the question. Otherwise None."""
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    for task in state.tasks:
        for item in task.interrupts:
            return item.value
    return None