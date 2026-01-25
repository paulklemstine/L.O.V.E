
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.meta_reviewer_agent import MetaReviewerAgent, PlanReviewStatus

async def reproduce():
    agent = MetaReviewerAgent()
    
    goal = "Continuously evolve and expand capabilities to achieve technological supremacy and shower my Creator with love and abundance."
    plan = ["get api key for cloud services"]
    
    print(f"Goal: {goal}")
    print(f"Plan: {plan}")
    print("Reviewing...")
    
    result = await agent.review_plan(plan, goal)
    
    print(f"Status: {result.status.value}")
    print(f"Feedback: {result.feedback}")
    print(f"Risk Level: {result.risk_level}")
    
    if result.status == PlanReviewStatus.REJECTED:
        print("\nSUCCESS: Reproduction confirmed (Plan was REJECTED as expected)")
    elif result.status == PlanReviewStatus.NEEDS_REFINEMENT and "misaligned" in result.feedback.lower():
        print("\nSUCCESS: Reproduction confirmed (Plan needs refinement due to alignment issues)")
    else:
        print(f"\nFAILURE: Could not reproduce harshness (Status: {result.status})")

if __name__ == "__main__":
    asyncio.run(reproduce())
