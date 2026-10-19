import ast
import operator
import requests
from ddgs import DDGS
from langchain_core.tools import tool


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


# --- safe calculator (never use eval on user text) ---
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
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


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city name, for example 'Delhi' or 'New York'."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=10,
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

        return (f"{place['name']}, {place.get('country', '')}: "
                f"{w['temperature_2m']}°C, humidity {w['relative_humidity_2m']}%, "
                f"wind {w['wind_speed_10m']} km/h")
    except Exception as e:
        return f"Weather lookup failed: {e}"


ALL_TOOLS = [web_search, calculator, get_weather]