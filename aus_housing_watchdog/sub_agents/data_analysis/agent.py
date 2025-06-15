import os
from datetime import date

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

from .prompts import return_instructions
from . import tools
import pathlib

date_today = date.today()

# Setup hooks for data analysis agent

def setup_before_agent_call(callback_context: CallbackContext):
    """Setup agent state if needed."""
    # Initialize analysis results dictionary
    callback_context.state["analysis_results"] = {}
    
    # Set path to processed data file
    data_dir = pathlib.Path.cwd() / "data"
    processed_data_path = data_dir / "processed-housing-data.csv"
    callback_context.state["processed_data_path"] = str(processed_data_path)
    
    # Set empty dictionaries for various types of analysis
    callback_context.state["price_trends"] = {}
    callback_context.state["location_analysis"] = {}
    callback_context.state["property_type_analysis"] = {}
    callback_context.state["market_anomalies"] = []
    callback_context.state["user_matching_results"] = {}


root_agent = Agent(
    model=os.getenv("SUB_AGENT_MODEL", "gemini-2.0-flash"),
    name="housing_market_analyzer",
    instruction=return_instructions(),
    global_instruction=(
        f"""
        You are the NSW Housing Market Analysis Agent.
        Your purpose is to analyze NSW housing market data to identify trends,
        opportunities, and risks based on the user's profile and requirements.
        Today's date: {date_today}
        """
    ),
    tools=[
        tools.run_comprehensive_analysis
    ],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
)
