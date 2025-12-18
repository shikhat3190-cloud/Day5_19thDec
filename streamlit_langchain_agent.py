import requests
import os
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# ----------------------------------
# 1. Weather API Tool
# ----------------------------------
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherInput(BaseModel):
    city: str = Field(..., description="Name of the city to get weather for")


@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """Get current weather for a given city."""
    print(f"Getting weather for {city}")
    r = requests.get(
        GEOCODING_API_URL,
        params={"name": city, "count": 1},
        timeout=15
    )
    r.raise_for_status()
    data = r.json()

    if not data.get("results"):
        return f"Could not find location for {city}"

    lat = data["results"][0]["latitude"]
    lon = data["results"][0]["longitude"]

    w = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True
        },
        timeout=15
    )
    w.raise_for_status()

    weather = w.json().get("current_weather")
    return (
        f"City: {city}\n"
        f"Latitude: {lat}, Longitude: {lon}\n"
        f"Weather: {weather}"
    )

# ----------------------------------
# 2. Tavily Search Tool
# ----------------------------------
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


class SearchInput(BaseModel):
    query: str = Field(..., description="Search query")


@tool(args_schema=SearchInput)
def tavily_search(query: str) -> str:
    """Search the web for recent and factual information."""
    print(f"Searching for {query}")
    response = tavily_client.search(
        query=query,
        max_results=5,
        include_answer=True
    )
    return response.get("answer", "No results found.")

# ----------------------------------
# 3. Chat Model
# ----------------------------------
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
)

# ----------------------------------
# 4. Create Agent (MULTI-TOOL)
# ----------------------------------
agent = create_agent(
    model=model,
    tools=[get_weather, tavily_search],
)

# ----------------------------------
# 5. Streamlit UI
# ----------------------------------
st.set_page_config(page_title="Agentic AI: Weather + Search", layout="centered")
st.title("=== Multi tool search AI (Weather + Tavily Search)")

user_input = st.text_input(
    "Ask a question",
    placeholder="What is the weather in Tokyo and any major news?"
)

if st.button("Run Agent"):
    if not user_input:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Agent thinking..."):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )

        st.subheader("🤖 Agent Response")
        st.write(response["messages"][-1].content)
