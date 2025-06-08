# aus_housing_watchdog/utils/utils.py

def init_user_profile(state: dict):
    """Initialize empty user profile if not present."""
    if "user_profile" not in state:
        state["user_profile"] = {
            "rent_or_buy": None,
            "salary": None,
            "family_size": None,
            "location_preferences": None,
            "budget": None,
            "special_preferences": None,
        }


def update_user_profile(state: dict, key: str, value):
    """Update a field in the user profile."""
    if "user_profile" not in state:
        init_user_profile(state)

    if key in state["user_profile"]:
        state["user_profile"][key] = value
    else:
        raise KeyError(f"Invalid user profile key: {key}")


def profile_is_complete(state: dict):
    """Check if user profile is complete enough to start analysis."""
    required_fields = ["rent_or_buy", "salary",
                       "family_size", "location_preferences", "budget"]

    for field in required_fields:
        if not state["user_profile"].get(field):
            return False
    return True


def summarize_user_profile(state: dict):
    """Return a text summary of the user profile."""
    profile = state["user_profile"]
    summary = f"""
    **User Profile:**

    - Rent or Buy: {profile.get('rent_or_buy')}
    - Salary: {profile.get('salary')}
    - Family Size: {profile.get('family_size')}
    - Location Preferences: {profile.get('location_preferences')}
    - Budget: {profile.get('budget')}
    - Special Preferences: {profile.get('special_preferences')}
    """
    return summary
