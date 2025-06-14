import os
from datetime import date

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

from .prompts import return_instructions
from .tools import (
    download_latest_data,
    extract_latest_data
)

date_today = date.today()

# Optional: add any setup hooks here

def setup_before_agent_call(callback_context: CallbackContext):
    """Setup agent state if needed."""
    # Example: init a result key
    callback_context.state[f"{__name__}_result"] = None

root_agent = Agent(
    model=os.getenv("SUB_AGENT_MODEL", "gemini-2.0-flash"),
    name="data_fetching_agent",
    instruction=return_instructions(),
    global_instruction=(
        f"""
        You are the {__name__} agent.
        Today's date: {date_today}
        """
    ),
    tools=[
        download_latest_data, 
        extract_latest_data
    ],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
