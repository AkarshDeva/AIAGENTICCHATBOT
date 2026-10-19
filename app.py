import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from chatbot import chatbot, get_text, get_all_threads

st.set_page_config(page_title="My Chatbot")


# ---------- small helper functions ----------

def make_thread_id():
    """Make a new unique chat name."""
    return str(uuid.uuid4())


def add_thread(thread_id):
    """Add the chat name to the list (only once)."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def new_chat():
    """Start a fresh chat."""
    new_id = make_thread_id()
    st.session_state["thread_id"] = new_id
    st.session_state["history"] = []
    add_thread(new_id)


def load_chat(thread_id):
    """Open an old chat from the database and rebuild its messages for the screen."""
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    saved_messages = state.values.get("messages", [])

    shown = []
    for m in saved_messages:
        if isinstance(m, HumanMessage):
            shown.append({"role": "user", "content": get_text(m.content)})
        elif isinstance(m, AIMessage):
            text = get_text(m.content)
            if text:                      # skip empty AI messages (tool requests)
                shown.append({"role": "assistant", "content": text})
    return shown


# ---------- first-time setup (runs once per browser tab) ----------

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = make_thread_id()

if "history" not in st.session_state:
    st.session_state["history"] = []

add_thread(st.session_state["thread_id"])


# ---------- sidebar ----------

st.sidebar.title("My Chats")

if st.sidebar.button("➕ New chat"):
    new_chat()

all_threads = st.session_state["chat_threads"]
for number, tid in reversed(list(enumerate(all_threads, start=1))):
    if st.sidebar.button(f"Chat {number}", key=tid):
        st.session_state["thread_id"] = tid
        st.session_state["history"] = load_chat(tid)


# ---------- main screen ----------

st.title("My Chatbot")

for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type here")

if user_input:
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    st.session_state["history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        status_box = {"box": None}

        def token_stream():
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(chunk, ToolMessage):
                    # a tool just finished: show a small status line
                    name = chunk.name or "tool"
                    if status_box["box"] is None:
                        status_box["box"] = st.status(f"Used tool: {name}", state="complete")
                    else:
                        status_box["box"].update(label=f"Used tool: {name}", state="complete")
                elif isinstance(chunk, AIMessage):
                    text = get_text(chunk.content)
                    if text:
                        yield text

        try:
            ai_text = st.write_stream(token_stream())
        except Exception:
            ai_text = "Sorry, something went wrong or the AI is busy. Please try again in a minute."
            st.write(ai_text)

    st.session_state["history"].append({"role": "assistant", "content": ai_text})