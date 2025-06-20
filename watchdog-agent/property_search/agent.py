from google.adk.agents import Agent
from google.adk.tools import google_search
from .prompts import return_instructions_root


root_agent = Agent(
    name="property_search",
    model="gemini-2.0-flash",
    description="Property search agent for Victorian properties",
    instruction=return_instructions_root(),
    global_instruction="You are a property search assistant for Victoria, Australia.",
    tools=[google_search]
)
