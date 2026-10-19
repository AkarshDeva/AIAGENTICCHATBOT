from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def text_of(content):
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str

def generate_joke(state: JokeState):
    r = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": text_of(r.content)}

def generate_explanation(state: JokeState):
    r = llm.invoke(f"Explain this joke briefly: {state['joke']}")
    return {"explanation": text_of(r.content)}

g = StateGraph(JokeState)
g.add_node("generate_joke", generate_joke)
g.add_node("generate_explanation", generate_explanation)
g.add_edge(START, "generate_joke")
g.add_edge("generate_joke", "generate_explanation")
g.add_edge("generate_explanation", END)

workflow = g.compile(checkpointer=InMemorySaver())

cfg1 = {"configurable": {"thread_id": "1"}}
cfg2 = {"configurable": {"thread_id": "2"}}

workflow.invoke({"topic": "football"}, config=cfg1)
workflow.invoke({"topic": "cricket"}, config=cfg2)

# 1. Final state of each thread (they are separate)
print("THREAD 1:", workflow.get_state(cfg1).values["topic"])
print("THREAD 2:", workflow.get_state(cfg2).values["topic"])

# 2. All snapshots of thread 1 (newest first)
history = list(workflow.get_state_history(cfg1))
print("\nNumber of checkpoints:", len(history))
for snap in history:
    v = snap.values
    print("next step:", snap.next,
          "| has topic:", "topic" in v,
          "| has joke:", "joke" in v,
          "| has explanation:", "explanation" in v)