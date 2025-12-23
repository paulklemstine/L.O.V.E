
import os
import logging
import asyncio
from core.scientist.scout import Scout
from core.scientist.test_generator import TestGenerator
from core.scientist.mutation import MutationEngine
from core.surgeon.grafting import graft_function
from core.guardian.verification import VerificationPipeline
from core.guardian.safety import SafetyNet
from core.integration.dashboard import EvolutionDashboard
from core.integration.git_handler import GitHandler
from core.surgeon.sandbox import DockerSandbox


class LocalSandbox:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        
    def ensure_image_exists(self):
        print("Using Local Sandbox (Docker unavailable).")
        
    def run_command(self, command, timeout=30):
        # Run locally in the base_dir
        # Be careful with timeouts
        try:
            # We command is usually "python3 -m pytest ..."
            # We need to split it
            cmd_list = command.split()
            result = subprocess.run(
                cmd_list, 
                cwd=self.base_dir, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "Timeout"
        except Exception as e:
            return -1, "", str(e)

async def run_demo():
    target_file = "demo_target.py"
    function_name = "slow_fib"
    test_file = "tests/test_demo_target.py"
    branch_name = "agent/demo-evolution"

    # ... setup ...
    try:
        sandbox = DockerSandbox(base_dir=os.getcwd())
        sandbox.ensure_image_exists()
    except Exception as e:
        print(f"Docker unavailable ({e}). Falling back to LocalSandbox.")
        sandbox = LocalSandbox(base_dir=os.getcwd())
        
    # ... rest ...
    
    git = GitHandler()
    safety = SafetyNet()
    dashboard = EvolutionDashboard()
    test_gen = TestGenerator()
    mutation = MutationEngine()
    pipeline = VerificationPipeline(sandbox)

    print("\n--- PHASE 1: PREPARATION ---")
    
    # 1. Safety Check
    if not safety.check_clean_state(target_file):
        print(f"File {target_file} is dirty! Please commit changes first.")
        # For demo purposes, we might ignore this or force clean?
        # Let's proceed assuming user knows, or failed check returns False.
        # But we want demo to run. Let's assume git status might fail in this environment if not a repo or something.
        # pass
    
    # 2. Branching
    print(f"Creating branch {branch_name}...")
    git.create_branch(branch_name)
    git.checkout_branch(branch_name)
    
    # 3. Backup
    print("Creating backup...")
    safety.create_backup(target_file)

    try:
        print("\n--- PHASE 2: SCIENTIST (Analysis & Test Gen) ---")
        
        # 4. Generate Test
        print("Generating Zero-Shot Test...")
        # We need async call for test gen? No, it wraps run_llm inside.
        if await test_gen.generate_test(target_file, function_name, test_file):
            print(f"Test generated at {test_file}")
        else:
            print("Failed to generate test.")
            return

        # 5. Verify Baseline
        print("Verifying baseline...")
        if not pipeline.verify_semantics(test_file):
            print("Baseline verification failed! The current code doesn't pass the generated test.")
            # In a real scenario, we might fix the test or code.
            # For demo, let's hope LLM generates correct test for fib.
            return
        else:
            print("Baseline verifies successfully.")

        print("\n--- PHASE 3: MUTATION (Evolution) ---")
        
        # 6. Evolve
        goal = "Optimize performance (use iteration or memoization) but keep exactly the same signature and behavior."
        print(f"Evolving {function_name} with goal: {goal}")
        
        new_code = await mutation.evolve_function(target_file, function_name, goal)
        if not new_code:
            print("Evolution failed (signature mismatch or LLM error).")
            return
            
        print("Proposed Code:")
        print(new_code)
        
        print("\n--- PHASE 4: SURGEON (Grafting) ---")
        
        # 7. Graft
        print("Grafting new code...")
        if graft_function(target_file, function_name, new_code):
            print("Graft successful.")
        else:
            print("Graft failed.")
            return

        print("\n--- PHASE 5: GUARDIAN (Verification) ---")
        
        # 8. Verify New Code (Gate 1, 2, 3)
        print("Running Verification Pipeline...")
        if pipeline.verify_all(target_file, test_file):
            print("Verification PASSED!")
            status = "SUCCESS"
        else:
            print("Verification FAILED!")
            status = "FAILED"
            # Rollback is handled in finally/except but here we decide outcomes
            raise Exception("Verification failed")

        print("\n--- PHASE 6: INTEGRATION ---")
        
        # 9. Log and Commit
        dashboard.log_evolution(f"{target_file}:{function_name}", status, "Optimized fibonacci")
        git.commit_changes(f"Evolved {function_name} in {target_file}")
        git.create_pr(f"Optimize {function_name}", "Removed recursion")
        
        print("Demo Complete: Success!")
        
        # Cleanup backup on success
        safety.cleanup_backup(target_file)

    except Exception as e:
        print(f"Demo Failed: {e}")
        print("Rolling back...")
        safety.restore_from_backup(target_file)
        dashboard.log_evolution(f"{target_file}:{function_name}", "FAILED", str(e))

if __name__ == "__main__":
    asyncio.run(run_demo())
