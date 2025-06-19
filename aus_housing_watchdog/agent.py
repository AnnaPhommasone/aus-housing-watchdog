from google.adk.agents import Agent
from .prompt import root_agent_instruction


def get_nsw_suburb_info(suburb_name: str, budget: int) -> dict:
    """
    Retrieves information about a specified NSW suburb based on a budget.

    Args:
        suburb_name (str): The name of the NSW suburb (e.g., "Chatswood", "Parramatta").
        budget (int): The user's budget.

    Returns:
        dict: A dictionary containing suburb information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with suburb details.
              If 'error', includes an 'error_message' key.
    """
    print(
        f"--- Tool: get_nsw_suburb_info called for suburb: {suburb_name} with budget: {budget} ---")
    suburb_normalized = suburb_name.lower().replace(" ", "")

    # Mock suburb data - in a real scenario, this would query the CSV or a database
    mock_suburb_db = {
        "chatswood": {
            "median_price": 1800000,
            "description": "Chatswood is a major commercial and residential hub with excellent transport links and shopping.",
            "recent_sales_within_budget": 5,
        },
        "parramatta": {
            "median_price": 900000,
            "description": "Parramatta is a growing CBD with diverse amenities and a rich history.",
            "recent_sales_within_budget": 12,
        },
        "bondi": {
            "median_price": 2500000,
            "description": "Bondi is famous for its beach, vibrant cafe culture, and coastal lifestyle.",
            "recent_sales_within_budget": 3,
        },
        "mosman": {
            "median_price": 4500000,
            "description": "Mosman is an affluent harbourside suburb known for its stunning views and luxury homes.",
            "recent_sales_within_budget": 2,
        },
        "penrith": {
            "median_price": 750000,
            "description": "Penrith is a developing regional city at the foot of the Blue Mountains, offering more affordable housing options.",
            "recent_sales_within_budget": 20,
        }
    }

    if suburb_normalized in mock_suburb_db:
        data = mock_suburb_db[suburb_normalized]
        # Simple check if budget is "reasonable" for the suburb's median price
        if budget >= data["median_price"] * 0.8 and budget <= data["median_price"] * 1.5:
            return {
                "status": "success",
                "report": (
                    f"Suburb: {suburb_name.title()}\n"
                    f"Description: {data['description']}\n"
                    f"Approx. Median Price: AUD ${data['median_price']:,}\n"
                    f"Recent sales potentially matching your budget of AUD ${budget:,}: {data['recent_sales_within_budget']}"
                )
            }
        else:
            return {
                "status": "success",
                "report": (
                    f"Suburb: {suburb_name.title()}\n"
                    f"Description: {data['description']}\n"
                    f"Approx. Median Price: AUD ${data['median_price']:,}\n"
                    f"Your budget of AUD ${budget:,} is significantly different from the median price. "
                    f"There were {data['recent_sales_within_budget']} recent sales in this suburb that might be closer to its typical price range."
                )
            }
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have information for the NSW suburb '{suburb_name}'."}


root_agent = Agent(
    name="root_agent_v1",
    model="gemini-2.0-flash",
    description="Root agent for the Aus Housing Watchdog system. Collects user budget and preferred suburbs, then provides information.",
    instruction=root_agent_instruction,
    tools=[get_nsw_suburb_info],
)
