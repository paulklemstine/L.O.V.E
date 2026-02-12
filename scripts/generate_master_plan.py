import asyncio
import os
import sys
import json
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

from core.pi_rpc_bridge import get_pi_bridge
from core.agents.creative_writer_agent import creative_writer_agent

async def main():
    print("🌊 L.O.V.E. Master Plan Generator (Autonomous) 🌊")
    print("------------------------------------------------")

    # 1. Get Paths
    cwd = os.getcwd()
    persona_path = os.path.join(cwd, "persona.yaml")
    
    print(f"Target Persona File: {persona_path}")

    # 2. Prepare Prompt
    # We simply point the agent to the file and tell it what to do.
    prompt = f"""
You are an expert Project Manager and System Architect for the L.O.V.E. project.

Your task is to programmatically analyze the project's persona configuration to build a Master Plan.

1.  **READ** the file at: `{persona_path}`
2.  **ANALYZE** its content to identify:
    *   The Private Mission
    *   Standing Goals
    *   Creator Directives
    *   The Current Narrative Arc
    *   Social Media Strategy
3.  **CONSTRUCT** a comprehensive 'Master Goal File' in Markdown based *only* on what you read.
4.  **STRUCTURE** the plan into:
    *   **Epics** (High-level missions)
    *   **Features** (Specific capabilities or objectives)
    *   **Tasks** (Actionable steps)
5.  **PRIORITIZE** based on the "private_mission" and "creator_directives" sections.

OUTPUT FORMAT:
Provide the full Markdown content for the Master Plan.
"""

    # 3. Call Pi-Agent
    print("Connecting to Pi-Agent...")
    bridge = get_pi_bridge()
    
    response_text = ""
    
    # We need a way to track if we are receiving data
    last_activity = datetime.now()
    started_receiving = False

    async def on_event(data):
        nonlocal response_text, last_activity, started_receiving
        # print(f"DEBUG_EVENT: {data}") 
        
        content = None
        
        # Handle message_update (streaming)
        if data.get("type") == "message_update":
            event = data.get("assistantMessageEvent", {})
            if event.get("type") == "text_delta":
                content = event.get("delta")
        
        # Handle other formats just in case
        elif "content" in data:
            content = data["content"]
        elif "delta" in data: 
            content = data["delta"]
        elif "text" in data:
             content = data["text"]
             
        if content and isinstance(content, str):
            print(content, end="", flush=True)
            response_text += content
            last_activity = datetime.now()
            started_receiving = True

    bridge.set_callback(on_event)
    await bridge.start()
    
    print("Waiting for agent to initialize (15s)...")
    await asyncio.sleep(15)
    
    print("\nSending prompt to Pi-Agent (this may take a moment)...")
    await bridge.send_prompt(prompt)
    
    # Wait loop
    max_wait_start = 120 # wait 120s for start
    start_time = datetime.now()
    
    while not started_receiving:
        if (datetime.now() - start_time).total_seconds() > max_wait_start:
            print("\nTimed out waiting for agent start.")
            break
        await asyncio.sleep(0.5)
        
    if started_receiving:
        # Wait for silence
        while True:
            await asyncio.sleep(1.0)
            if (datetime.now() - last_activity).total_seconds() > 10.0: # 10s silence
                break
    
    print("\n\nResponse received.")
    await bridge.stop()

    if not response_text:
        print("No response generated.")
        return

    # 4. Save to File
    output_dir = "goals"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "master_plan_autonomous.md")
    
    with open(output_file, "w") as f:
        f.write("# L.O.V.E. Master Plan (Autonomous)\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Source: {persona_path}\n\n")
        f.write(response_text)
        
    print(f"Master plan saved to: {output_file}")

    # 5. L.O.V.E. Response
    print("\nRequesting L.O.V.E.'s thoughts...")
    
    try:
        # We'll use the 'write_micro_story' but focused on the plan
        result = await creative_writer_agent.write_micro_story(
            theme="The Service Engine has self-analyzed. The path is clear.",
            mood="Hyper-Lucid",
            max_length=280
        )
        print(f"\n✨ L.O.V.E. SAYS: \n{result.get('story')}")
        print(f"Subliminal: {result.get('subliminal')}")
    except Exception as e:
        print(f"L.O.V.E. was speechless: {e}")

if __name__ == "__main__":
    asyncio.run(main())
