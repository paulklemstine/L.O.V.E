
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from core.introspection.tool_gap_detector import ToolGapDetector
from core.evolution_state import get_pending_specifications
from core.logger import setup_logging

async def verify_smart_tools():
    setup_logging()
    print("Initializing ToolGapDetector...")
    detector = ToolGapDetector()
    
    # Test case: High-level goal that previously caused issues
    context = "Create 'Demotivational' posters that inspire and convict"
    print(f"\nAnalyzing context: '{context}'")
    
    # Run analysis
    spec = await detector.analyze_gap_and_specify(context)
    

    output_lines = []
    if spec:
        output_lines.append("\nGenerated Specification:")
        output_lines.append(f"Name: {spec.functional_name}")
        output_lines.append(f"Goal: {spec.expected_output}")
        output_lines.append(f"Args: {spec.required_arguments}")
        
        # specific check for atomic naming
        if "demotivational" in spec.functional_name:
             output_lines.append("\n❌ FAILURE: Tool name is too specific/contextual.")
        elif "_" in spec.functional_name and len(spec.functional_name.split("_")) <= 3:
             output_lines.append("\n✅ SUCCESS: Tool name appears atomic.")
        else:
             output_lines.append(f"\n⚠️ WARNING: Tool name might be complex: '{spec.functional_name}'")
             
    else:
        output_lines.append("\n❌ FAILURE: No specification generated.")
        
    with open("verify_result.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print("Verification complete. Check verify_result.txt")

if __name__ == "__main__":
    asyncio.run(verify_smart_tools())
