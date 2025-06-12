import os
from datetime import date
from .utils.utils import init_user_profile

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

from . import prompts
from . import tools
from .sub_agents import (
    cleaning_data,
    data_analysis,
    recommendation,
    visualiser,
)

date_today = date.today()


def setup_before_agent_call(callback_context: CallbackContext):
    """Setup the housing watchdog agent."""

    callback_context.state["project_id"] = os.getenv("GOOGLE_CLOUD_PROJECT")
    callback_context.state["bq_dataset"] = os.getenv("BQ_DATASET_ID")
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
        You orchestrate sub-agents to perform various tasks like data cleaning, analysis, anomaly detection, and report generation.
        Today's date: {date_today}
        """
    ),
    sub_agents=[],   # ⛔️ temporarily empty
    tools=[],        # ⛔️ temporarily empty
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
