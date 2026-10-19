import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from chatbot import chatbot, get_text, get_all_threads, get_pending_approval

st.set_page_config(page_title="My Chatbot")


# ---------- helper functions ----------

def make_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def new_chat():
    new_id = make_thread_id()
    st.session_state["thread_id"] = new_id
    st.session_state["history"] = []
    add_thread(new_id)


def load_chat(thread_id):
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    shown = []
    for m in state.values.get("messages", []):
        if isinstance(m, HumanMessage):
            shown.append({"role": "user", "content": get_text(m.content)})
        elif isinstance(m, AIMessage):
            text = get_text(m.content)
            if text:
                shown.append({"role": "assistant", "content": text})
    return shown


def token_stream(payload, config):
    """Run the bot and hand out the answer piece by piece.
    payload = a new message, OR Command(resume=...) to continue after a pause."""
    for chunk, metadata in chatbot.stream(payload, config=config, stream_mode="messages"):
        if isinstance(chunk, AIMessage):
            text = get_text(chunk.content)
            if text:
                yield text


# ---------- first-time setup ----------

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

for number, tid in reversed(list(enumerate(st.session_state["chat_threads"], start=1))):
    if st.sidebar.button(f"Chat {number}", key=tid):
        st.session_state["thread_id"] = tid
        st.session_state["history"] = load_chat(tid)

# (Add the "Upload a PDF" sidebar block from Lesson 20 here if you want RAG too.)


# ---------- main screen ----------

st.title("My Chatbot")

for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

thread_id = st.session_state["thread_id"]
CONFIG = {
    "configurable": {"thread_id": thread_id},
    "metadata": {"thread_id": thread_id},
    "run_name": "chat_turn",
}

# ----- Is this chat paused and waiting for a human decision? -----
pending = get_pending_approval(thread_id)

if pending:
    st.warning(f"⚠️ Approval needed: {pending}")
    col1, col2 = st.columns(2)
    answer = None
    if col1.button("✅ Approve"):
        answer = "yes"
    if col2.button("❌ Reject"):
        answer = "no"

    if answer:
        with st.chat_message("assistant"):
            ai_text = st.write_stream(token_stream(Command(resume=answer), CONFIG))
        if ai_text:
            st.session_state["history"].append({"role": "assistant", "content": ai_text})
        st.rerun()

    st.stop()   # no typing box until the human decides


# ----- Normal chat -----
user_input = st.chat_input("Type here")

if user_input:
    st.session_state["history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        try:
            ai_text = st.write_stream(
                token_stream({"messages": [HumanMessage(content=user_input)]}, CONFIG)
            )
        except Exception:
            ai_text = "Sorry, something went wrong or the AI is busy. Please try again in a minute."
            st.write(ai_text)

    if ai_text:
        st.session_state["history"].append({"role": "assistant", "content": ai_text})

    # did the bot pause? then reload the page so the Approve/Reject buttons show
    if get_pending_approval(thread_id):
        st.rerun()