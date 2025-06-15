import os
from datetime import date

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

from .prompts import return_instructions
from .tools import (
    load_raw_data,
    process_clean_data
)

date_today = date.today()

# Optional: add any setup hooks here

def setup_before_agent_call(callback_context: CallbackContext):
    """Setup agent state if needed."""
    # Initialize a result key for this agent
    callback_context.state[f"{__name__}_result"] = None
    # Set the path for processed data
    callback_context.state["processed_data_path"] = "./data/processed-housing-data.csv"

root_agent = Agent(
    model=os.getenv("SUB_AGENT_MODEL", "gemini-2.0-flash"),
    name="data_processing_agent",
    instruction=return_instructions(),
    global_instruction=(
        f"""
        You are the {__name__} agent, responsible for preparing housing data for analysis.
        Today's date: {date_today}
        """
    ),
    tools=[
        load_raw_data, 
        process_clean_data
    ],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
