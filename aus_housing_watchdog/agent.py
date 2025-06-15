import os
from datetime import date
from .utils.utils import init_user_profile

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

from . import prompts
from .sub_agents import (
    data_fetch_and_clean,
    data_analysis,
    recommendation,
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
        You are a multi-agent system monitoring and analyzing the NSW housing market in Australia.
        You orchestrate sub-agents to process housing data, analyze it, and generate recommendations.
        You have three sub-agents available:
        1. Data Processing Agent: Loads and cleans local NSW housing data
        2. Data Analysis Agent: Analyzes the cleaned data and identifies trends, patterns, and insights
        3. Recommendation Agent: Generates personalized property and suburb recommendations based on analysis
        Today's date: {date_today}
        """
    ),
    sub_agents=[
        data_fetch_and_clean.agent,
        data_analysis.agent,
        recommendation.agent
    ],
    tools=[],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
