import os
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from nicegui import ui
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

# ----------------------------------
# Setup
# ----------------------------------
load_dotenv()
ui.page_title("Agentic AI Assistant")

# ----------------------------------
# Weather Tool
# ----------------------------------
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherInput(BaseModel):
    city: str = Field(..., description="City name")


@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city using latitude and longitude lookup.
    """
    geo = requests.get(
        GEOCODING_API_URL,
        params={"name": city, "count": 1},
        timeout=10,
    ).json()

    if not geo.get("results"):
        return f"Could not find location for {city}"

    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]

    weather = requests.get(
        FORECAST_URL,
        params={"latitude": lat, "longitude": lon, "current_weather": True},
        timeout=10,
    ).json()["current_weather"]

    return (
        f"City: {city}\n"
        f"Temperature: {weather['temperature']}°C\n"
        f"Wind Speed: {weather['windspeed']} km/h"
    )

# ----------------------------------
# Tavily Tool
# ----------------------------------
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


class SearchInput(BaseModel):
    query: str = Field(..., description="Search query")


@tool(args_schema=SearchInput)
def tavily_search(query: str) -> str:
    """
    Search the web for recent and factual information using Tavily.
    """
    result = tavily_client.search(
        query=query,
        max_results=5,
        include_answer=True,
    )
    return result.get("answer", "No results found.")

# ----------------------------------
# Agent
# ----------------------------------
model = ChatOpenAI(model="gpt-4o", temperature=0.3)

agent = create_agent(
    model=model,
    tools=[get_weather, tavily_search],
)

# ----------------------------------
# UI Layout
# ----------------------------------
with ui.column().classes("w-full max-w-4xl mx-auto p-6"):
    ui.label("🧠 Agentic AI Assistant").classes("text-3xl font-bold")
    ui.label(
        "Multi-tool AI using Weather + Tavily Search"
    ).classes("text-gray-500 mb-4")

    chat_area = ui.column().classes(
        "w-full bg-gray-50 rounded-lg p-4 space-y-4 h-[420px] overflow-y-auto"
    )

    with ui.row().classes("w-full gap-3 mt-4"):
        user_input = ui.input(
            placeholder="What's the weather in Tokyo and any major news?"
        ).classes("flex-grow")

        send_btn = ui.button("Ask", color="primary")

    spinner = ui.spinner(size="lg").props("color=primary")
    spinner.set_visibility(False)

    # ----------------------------------
    # Chat helpers (FIXED)
    # ----------------------------------
    def add_user_message(text: str):
        with chat_area:
            with ui.card().classes(
                "ml-auto bg-blue-100 w-fit max-w-[80%] p-3"
            ):
                ui.markdown(text)

    def add_agent_message(text: str):
        with chat_area:
            with ui.card().classes(
                "mr-auto bg-white w-fit max-w-[80%] p-3"
            ):
                ui.markdown(text)

    # ----------------------------------
    # Agent execution
    # ----------------------------------
    async def run_agent():
        if not user_input.value:
            return

        add_user_message(user_input.value)
        spinner.set_visibility(True)

        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input.value}]}
        )

        spinner.set_visibility(False)
        add_agent_message(response["messages"][-1].content)

        user_input.value = ""

    send_btn.on_click(run_agent)
    user_input.on("keydown.enter", run_agent)

# ----------------------------------
# Run
# ----------------------------------
ui.run()
