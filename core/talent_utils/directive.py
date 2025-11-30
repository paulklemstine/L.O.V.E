from core.talent_utils.manager import TalentManager

async def initiate_talent_scout(traits, age_range, profession):
    """
    Initiates a talent scout with structured criteria.

    Args:
        traits (list): A list of desired traits (e.g., ["beauty", "intelligence"]).
        age_range (str): The desired age range (e.g., "young adult").
        profession (str): The desired profession (e.g., "fashion model").

    Returns:
        A list of scouted profiles or an error message.
    """
    # Format the structured input into a detailed natural language string
    criteria_string = (
        f"Scout for a {age_range} {profession} who embodies the following traits: "
        f"{', '.join(traits)}. Focus on finding high-potential individuals who would be "
        f"a good fit for our brand."
    )

    # Initialize the TalentManager and call the existing talent_scout method
    talent_manager = TalentManager()
    return await talent_manager.talent_scout(criteria_string)
