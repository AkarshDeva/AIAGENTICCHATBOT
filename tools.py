import ast
import operator
import requests
from ddgs import DDGS
from langchain_core.tools import tool
from langgraph.types import interrupt

from rag import search_pdf


# ---------- Tool 1: internet search ----------
@tool
def web_search(query: str) -> str:
    """Search the internet. Use for latest news, recent events, or facts you are not sure about."""
    try:
        results = DDGS().text(query, max_results=5)
        return "\n\n".join(
            f"Title: {r['title']}\nLink: {r['href']}\nSummary: {r['body']}"
            for r in results
        )
    except Exception as e:
        return f"Search failed: {e}"


# ---------- Tool 2: safe calculator ----------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _calc(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_calc(node.left), _calc(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_calc(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Calculate a math expression like '25/100*8500' or '(12+8)*3'. Use for any arithmetic."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_calc(tree.body))
    except Exception as e:
        return f"Could not calculate: {e}"


# ---------- Tool 3: weather (free, no key) ----------
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city name, for example 'Delhi' or 'New York'."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        if not geo.get("results"):
            return f"Could not find the city '{city}'."
        place = geo["results"][0]

        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            },
            timeout=10,
        ).json()["current"]

        return (
            f"{place['name']}, {place.get('country', '')}: "
            f"{w['temperature_2m']}°C, humidity {w['relative_humidity_2m']}%, "
            f"wind {w['wind_speed_10m']} km/h"
        )
    except Exception as e:
        return f"Weather lookup failed: {e}"


# ---------- Tool 4: search the uploaded PDFs (RAG) ----------
@tool
def search_documents(question: str) -> str:
    """Search the user's uploaded PDF documents. Use this for any question about an uploaded PDF, paper or document."""
    try:
        docs = search_pdf(question)
        if docs is None:
            return "No document has been uploaded yet. Ask the user to upload a PDF."
        if not docs:
            return "No relevant information was found in the uploaded documents."
        parts = []
        for d in docs:
            page = d.metadata.get("page", "?")
            parts.append(f"[Page {page}] {d.page_content}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Document search failed: {e}"


# ---------- Tool 5: risky action, needs human approval ----------
@tool
def purchase_stock(symbol: str, quantity: int) -> str:
    """Buy shares of a company's stock. Use ONLY when the user clearly asks to buy or purchase stock."""
    # PAUSE here and ask the human. Nothing below runs until they answer.
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}?")

    if str(decision).lower() == "yes":
        # a real app would call a broker API here
        return f"Order placed: {quantity} shares of {symbol} (demo only, no real money)."
    return f"The purchase of {quantity} shares of {symbol} was declined by the user."


# ---------- The toolbox given to the AI ----------
ALL_TOOLS = [web_search, calculator, get_weather, search_documents, purchase_stock]