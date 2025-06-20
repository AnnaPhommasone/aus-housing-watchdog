from google.adk.agents import Agent

root_agent = Agent(
    name="property_search",
    model="gemini-2.0-flash",
    description="Greeting property agent",
    instruction="""
    You are a helpful assistant that greets the user. Ask for user's name and where they would like to buy a property in Australia.
    """
)
