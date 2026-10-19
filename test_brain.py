from langchain_core.messages import HumanMessage
from chatbot import chatbot, get_text

config = {"configurable": {"thread_id": "1"}}

while True:
    text = input("You: ")
    if text.lower() in ["exit", "quit", "bye"]:
        break
    try:
        result = chatbot.invoke({"messages": [HumanMessage(content=text)]}, config=config)
        print("Bot:", get_text(result["messages"][-1].content))
    except Exception as e:
        print("ERROR TYPE:", type(e).__name__)
        print("ERROR MESSAGE:", str(e)[:500])