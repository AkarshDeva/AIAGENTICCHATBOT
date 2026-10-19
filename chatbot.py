from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

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


# 3. THE STATE: a list of messages that grows
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 4. THE NODE: send the full history to the model, return its reply
def chat_node(state: ChatState):
    reply = llm.invoke(state["messages"])
    return {"messages": [reply]}


# 5. THE MEMORY (RAM only, erased when the program stops)
checkpointer = MemorySaver()

# 6. THE GRAPH
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)