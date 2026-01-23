#!/usr/bin/env python3
"""
run.py - L.O.V.E. Version 2 Entry Point

Starts the DeepLoop autonomous goal engine.

Usage:
    python run.py              # Run continuous loop
    python run.py --test-mode  # Run 3 iterations only
    python run.py --sleep 60   # Set sleep interval
"""

import os
import sys
from pathlib import Path

# Get the love2 directory and L.O.V.E. root
LOVE2_DIR = Path(__file__).parent.absolute()
LOVE_ROOT = LOVE2_DIR.parent

# CRITICAL: Remove any existing 'core' from sys.modules to avoid conflict
# with L.O.V.E. v1's core module
for key in list(sys.modules.keys()):
    if key.startswith('core'):
        del sys.modules[key]

# Add love2 directory FIRST so its 'core' takes precedence
sys.path.insert(0, str(LOVE2_DIR))
# Add L.O.V.E. root AFTER for v1 fallback imports (used by tool_adapter)
sys.path.insert(1, str(LOVE_ROOT))

# Change working directory to love2
os.chdir(LOVE2_DIR)

# Load environment
from dotenv import load_dotenv
load_dotenv(LOVE2_DIR / ".env")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="L.O.V.E. v2 DeepLoop - Autonomous Goal Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                    # Run forever
    python run.py --test-mode        # Run 3 iterations
    python run.py --sleep 60         # 60 second intervals
    python run.py --iterations 10    # Run exactly 10 iterations
        """
    )
    
    parser.add_argument(
        "--test-mode", "-t",
        action="store_true",
        help="Run only 3 iterations (for testing)"
    )
    
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=None,
        help="Number of iterations to run (default: infinite)"
    )
    
    parser.add_argument(
        "--sleep", "-s",
        type=float,
        default=0,
        help="Seconds between iterations (default: 0)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Determine iteration count
    max_iterations = None
    if args.test_mode:
        max_iterations = 3
    elif args.iterations:
        max_iterations = args.iterations
    
    # Print banner
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🌊 L.O.V.E. VERSION 2 - DeepAgent Autonomous Engine 🌊    ║
║                                                              ║
║   Living Organism, Vast Empathy                              ║
║   Unified Agentic Reasoning • Brain-Inspired Memory         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Import and run - love2/core is now first in sys.path
    from core.deep_loop import DeepLoop
    from core.logger import setup_logging
    from core.web_server import start_background_server
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Start Web Control Panel
    start_background_server()
    print("\n[Control Panel] 🌐 Web UI running at http://localhost:8000")
    
    loop = DeepLoop(
        max_iterations=max_iterations,
        sleep_seconds=args.sleep
    )
    
    print(f"Configuration:")
    print(f"  • Iterations: {'Infinite' if max_iterations is None else max_iterations}")
    print(f"  • Sleep interval: {args.sleep}s")
    print(f"  • Tools loaded: {len(loop.tools)}")
    print()
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[run.py] Interrupted by user")
        loop.stop()
    
    print("\n[run.py] Goodbye! 🌊")


if __name__ == "__main__":
    main()
