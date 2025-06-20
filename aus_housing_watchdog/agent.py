import os
from datetime import date
from .utils.utils import init_user_profile

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

from . import prompts
from .sub_agents import (
    data_fetching,
)

date_today = date.today()


def setup_before_agent_call(callback_context: CallbackContext):
    """Setup the housing watchdog agent."""

    callback_context.state["project_id"] = os.getenv("GOOGLE_CLOUD_PROJECT")
    # callback_context.state["bq_dataset"] = os.getenv("BQ_DATASET_ID")
    # Initialise user profile memory
    init_user_profile(callback_context.state)

    # Optionally insert known BQ schema if you generate it
    # callback_context.state["housing_data_schema"] = get_bq_housing_schema()

    print(
        f"[INFO] Root agent context initialized for project {callback_context.state['project_id']}")


root_agent = Agent(
    model=os.getenv("ROOT_AGENT_MODEL"),
    name="aus_housing_watchdog_root_agent",
    instruction=prompts.return_instructions_root(),
    global_instruction=(
        f"""
        You are a multi-agent system monitoring and analyzing the Australian housing market.
        You orchestrate sub-agents to process and prepare housing data for analysis.
        Currently, only the data processing agent is available to load and clean local housing data.
        Today's date: {date_today}
        """
    ),
    sub_agents=[
        data_fetching.agent
    ],
    tools=[],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
