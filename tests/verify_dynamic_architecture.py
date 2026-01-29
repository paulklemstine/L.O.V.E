
"""
Verification script for Open Agentic Web components.
Checks if all new modules can be imported and initialized.
"""
import sys
import os
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_import(module_name):
    try:
        __import__(module_name)
        logger.info(f"✅ Import successful: {module_name}")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {module_name} - {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error importing {module_name}: {e}")
        return False

async def verify_components():
    print(f"python version: {sys.version}")
    print(f"cwd: {os.getcwd()}")
    
    # Add current directory to path
    sys.path.append(os.getcwd())
    
    success = True
    
    # 1. Check Imports
    modules = [
        "core.codeact_engine",
        "core.mcp_registry",
        "core.docker_sandbox",
        "core.tool_retriever",
        "core.agents.evolutionary_agent",
        "core.dynamic_tools"
    ]
    
    for mod in modules:
        if not check_import(mod):
            success = False
            
    if not success:
        return False
        
    # 2. Check Initialization
    try:
        from core.mcp_registry import MCPRegistry
        registry = MCPRegistry()
        logger.info("✅ MCPRegistry initialized")
    except Exception as e:
        logger.error(f"❌ MCPRegistry init failed: {e}")
        success = False

    try:
        from core.codeact_engine import CodeActEngine
        engine = CodeActEngine()
        logger.info("✅ CodeActEngine initialized")
    except Exception as e:
        logger.error(f"❌ CodeActEngine init failed: {e}")
        success = False
        
    try:
        from core.docker_sandbox import UnifiedSandbox
        sandbox = UnifiedSandbox()
        available = sandbox.is_docker_available()
        logger.info(f"✅ UnifiedSandbox initialized (Docker available: {available})")
    except Exception as e:
        logger.error(f"❌ UnifiedSandbox init failed: {e}")
        success = False
        
    try:
        from core.dynamic_tools import search_mcp_servers
        # Just check it has the schema
        if hasattr(search_mcp_servers, "__tool_schema__"):
             logger.info("✅ core.dynamic_tools loaded correctly")
        else:
             logger.error("❌ core.dynamic_tools missing schema")
             success = False
    except Exception as e:
        logger.error(f"❌ dynamic_tools check failed: {e}")
        success = False

    return success

if __name__ == "__main__":
    try:
        result = asyncio.run(verify_components())
        if result:
            print("\n✨ All Open Agentic Web components verified successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Verification finished with errors.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
