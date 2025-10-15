class AutomatedBenchmarker:
    def run_benchmark(self, new_code, experiment_plan):
        """
        Simulates running a benchmark to compare the performance of the old and new code.
        """
        print("AutomatedBenchmarker: Running benchmark...")
        # This simulation will always return a positive result if the plan is correct.
        if experiment_plan.get("metric") == "token_usage" and "Web Search" in experiment_plan.get("name"):
            benchmark_result = {
                "control_performance": {"token_usage": 1500},
                "variant_performance": {"token_usage": 300},
                "conclusion": "Hypothesis confirmed: Token usage decreased by 80%."
            }
            print(f"AutomatedBenchmarker: Benchmark complete. {benchmark_result['conclusion']}")
            return True, benchmark_result
        else:
            print("AutomatedBenchmarker: Benchmark failed - invalid experiment plan.")
            return False, None