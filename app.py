import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot import chatbot, get_text

st.title("My Chatbot")

CONFIG = {"configurable": {"thread_id": "thread-1"}}

# session_state survives Streamlit re-runs; it stores what is SHOWN on screen
if "history" not in st.session_state:
    st.session_state["history"] = []

# Redraw old messages
for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type here")

if user_input:
    # show and save the user's message
    st.session_state["history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # stream the AI's answer
    with st.chat_message("assistant"):

        def token_stream():
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                text = get_text(chunk.content)
                if text:
                    yield text

        ai_text = st.write_stream(token_stream())

    st.session_state["history"].append({"role": "assistant", "content": ai_text})