"""
Model Quality Controller

Tracks performance metrics for OpenRouter and Horde models and automatically
blacklists models that consistently fail or produce poor results.

Quality Metrics:
- failure_rate: Ratio of failed calls to total calls
- consecutive_failures: Streak of failures (for rapid blacklisting)
- total_calls: Number of attempts made

Blacklist Thresholds:
- Failure Rate: >=70% after >=5 calls
- Consecutive Failures: >=5 consecutive failures
- HTTP 404 Errors: >=5 (model doesn't exist)
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Optional, Any
from collections import defaultdict
from core.logging import log_event

# --- Configuration ---
FAILURE_RATE_THRESHOLD = 0.70  # 70% failure rate
MIN_CALLS_FOR_RATE_CHECK = 5   # Minimum calls before evaluating failure rate
CONSECUTIVE_FAILURE_THRESHOLD = 5  # Consecutive failures before blacklist
HTTP_404_THRESHOLD = 5  # 404 errors before blacklist (model doesn't exist)

# Providers that are tracked for quality control
TRACKED_PROVIDERS = ["openrouter", "horde"]

# Path to auto-blacklist file
AUTO_BLACKLIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_model_blacklist.json")


def _create_default_stats() -> Dict[str, Any]:
    """Create default quality stats for a new model."""
    return {
        "total_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "consecutive_failures": 0,
        "http_404_count": 0,
        "last_success_time": None,
        "last_failure_time": None,
        "blacklisted": False,
        "blacklist_reason": None,
        "blacklisted_at": None,
        "provider": "unknown",
    }


class ModelQualityController:
    """
    Manages quality tracking and automatic blacklisting for LLM models.
    
    Tracks failure rates and consecutive failures for OpenRouter and Horde
    models, automatically blacklisting poor performers to improve system
    reliability.
    """
    
    def __init__(self):
        self._quality_stats: Dict[str, Dict[str, Any]] = defaultdict(_create_default_stats)
        self._auto_blacklist: Dict[str, Dict[str, Any]] = {}
        self._load_blacklist()
    
    def _get_full_model_id(self, model_id: str, provider: str) -> str:
        """Generate consistent full model ID for tracking."""
        if model_id.startswith(f"{provider}:"):
            return model_id
        return f"{provider}:{model_id}"
    
    def _load_blacklist(self) -> None:
        """Load auto-blacklist from persistent storage."""
        if os.path.exists(AUTO_BLACKLIST_PATH):
            try:
                with open(AUTO_BLACKLIST_PATH, "r") as f:
                    self._auto_blacklist = json.load(f)
                log_event(f"Loaded {len(self._auto_blacklist)} auto-blacklisted models.", "INFO")
            except (json.JSONDecodeError, IOError) as e:
                log_event(f"Error loading auto_model_blacklist.json: {e}", "WARNING")
                self._auto_blacklist = {}
        else:
            self._auto_blacklist = {}
    
    def _save_blacklist(self) -> None:
        """Save auto-blacklist to persistent storage."""
        try:
            with open(AUTO_BLACKLIST_PATH, "w") as f:
                json.dump(self._auto_blacklist, f, indent=2)
            log_event(f"Saved auto-blacklist with {len(self._auto_blacklist)} models.", "DEBUG")
        except IOError as e:
            log_event(f"Error saving auto_model_blacklist.json: {e}", "ERROR")
    
    def _evaluate_blacklist(self, full_model_id: str) -> Optional[str]:
        """
        Evaluate if a model should be blacklisted based on its stats.
        
        Returns:
            Blacklist reason if model should be blacklisted, None otherwise.
        """
        stats = self._quality_stats[full_model_id]
        
        # Check consecutive failures threshold
        if stats["consecutive_failures"] >= CONSECUTIVE_FAILURE_THRESHOLD:
            return f"consecutive_failures_exceeded ({stats['consecutive_failures']} failures)"
        
        # Check HTTP 404 errors (model doesn't exist)
        if stats["http_404_count"] >= HTTP_404_THRESHOLD:
            return f"http_404_errors ({stats['http_404_count']} errors - model may not exist)"
        
        # Check failure rate after minimum calls
        if stats["total_calls"] >= MIN_CALLS_FOR_RATE_CHECK:
            failure_rate = stats["failed_calls"] / stats["total_calls"]
            if failure_rate >= FAILURE_RATE_THRESHOLD:
                return f"failure_rate_exceeded ({failure_rate:.1%} after {stats['total_calls']} calls)"
        
        return None
    
    def _blacklist_model(self, full_model_id: str, reason: str) -> None:
        """Add a model to the auto-blacklist."""
        stats = self._quality_stats[full_model_id]
        
        # Update stats
        stats["blacklisted"] = True
        stats["blacklist_reason"] = reason
        stats["blacklisted_at"] = datetime.utcnow().isoformat() + "Z"
        
        # Add to persistent blacklist
        failure_rate = 0.0
        if stats["total_calls"] > 0:
            failure_rate = stats["failed_calls"] / stats["total_calls"]
        
        self._auto_blacklist[full_model_id] = {
            "blacklisted": True,
            "reason": reason,
            "failure_rate": round(failure_rate, 3),
            "total_calls": stats["total_calls"],
            "blacklisted_at": stats["blacklisted_at"],
            "provider": stats["provider"],
        }
        
        self._save_blacklist()
        log_event(f"AUTO-BLACKLISTED model '{full_model_id}': {reason}", "WARNING")
    
    def record_success(self, model_id: str, provider: str) -> None:
        """
        Record a successful call for a model.
        
        Resets consecutive failure count but preserves total counts for
        accurate failure rate calculation.
        """
        if provider not in TRACKED_PROVIDERS:
            return
        
        full_model_id = self._get_full_model_id(model_id, provider)
        stats = self._quality_stats[full_model_id]
        
        stats["total_calls"] += 1
        stats["successful_calls"] += 1
        stats["consecutive_failures"] = 0  # Reset on success
        stats["last_success_time"] = time.time()
        stats["provider"] = provider
        
        log_event(f"Quality: Recorded success for '{full_model_id}' ({stats['successful_calls']}/{stats['total_calls']})", "DEBUG")
    
    def record_failure(self, model_id: str, provider: str, error_type: str = "general") -> bool:
        """
        Record a failed call for a model and evaluate for blacklisting.
        
        Args:
            model_id: The model identifier
            provider: The provider (openrouter, horde)
            error_type: Type of error (general, http_404, timeout, etc.)
        
        Returns:
            True if the model was blacklisted as a result of this failure.
        """
        if provider not in TRACKED_PROVIDERS:
            return False
        
        full_model_id = self._get_full_model_id(model_id, provider)
        stats = self._quality_stats[full_model_id]
        
        # Update failure stats
        stats["total_calls"] += 1
        stats["failed_calls"] += 1
        stats["consecutive_failures"] += 1
        stats["last_failure_time"] = time.time()
        stats["provider"] = provider
        
        # Track specific error types
        if error_type == "http_404":
            stats["http_404_count"] = stats.get("http_404_count", 0) + 1
        
        failure_rate = stats["failed_calls"] / stats["total_calls"] if stats["total_calls"] > 0 else 0
        log_event(
            f"Quality: Recorded failure for '{full_model_id}' "
            f"(failures: {stats['failed_calls']}/{stats['total_calls']}, "
            f"consecutive: {stats['consecutive_failures']}, "
            f"rate: {failure_rate:.1%})",
            "DEBUG"
        )
        
        # Evaluate for blacklisting
        if not stats["blacklisted"]:
            reason = self._evaluate_blacklist(full_model_id)
            if reason:
                self._blacklist_model(full_model_id, reason)
                return True
        
        return False
    
    def is_blacklisted(self, model_id: str, provider: str = None) -> bool:
        """
        Check if a model is in the auto-blacklist.
        
        Args:
            model_id: The model identifier (can be full or short form)
            provider: Optional provider for short-form model_id
        
        Returns:
            True if the model is blacklisted.
        """
        # Handle full model IDs (e.g., "openrouter:model:free")
        if ":" in model_id and provider is None:
            return model_id in self._auto_blacklist
        
        # Handle short model IDs with provider
        if provider:
            full_model_id = self._get_full_model_id(model_id, provider)
            return full_model_id in self._auto_blacklist
        
        # Check all providers for this model
        for p in TRACKED_PROVIDERS:
            full_model_id = self._get_full_model_id(model_id, p)
            if full_model_id in self._auto_blacklist:
                return True
        
        return False
    
    def get_model_stats(self, model_id: str, provider: str) -> Dict[str, Any]:
        """Get quality statistics for a specific model."""
        full_model_id = self._get_full_model_id(model_id, provider)
        return dict(self._quality_stats[full_model_id])
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get quality statistics for all tracked models."""
        return dict(self._quality_stats)
    
    def get_blacklist(self) -> Dict[str, Dict[str, Any]]:
        """Get the current auto-blacklist."""
        return dict(self._auto_blacklist)
    
    def get_blacklist_summary(self) -> str:
        """Get a human-readable summary of blacklisted models."""
        if not self._auto_blacklist:
            return "No models are currently auto-blacklisted."
        
        lines = [f"Auto-Blacklisted Models ({len(self._auto_blacklist)}):"]
        for model_id, info in self._auto_blacklist.items():
            lines.append(
                f"  - {model_id}: {info.get('reason', 'Unknown')} "
                f"(rate: {info.get('failure_rate', 0):.1%}, calls: {info.get('total_calls', 0)})"
            )
        return "\n".join(lines)
    
    def remove_from_blacklist(self, model_id: str, provider: str = None) -> bool:
        """
        Remove a model from the auto-blacklist (manual override).
        
        Returns:
            True if the model was removed, False if it wasn't blacklisted.
        """
        full_model_id = model_id
        if provider:
            full_model_id = self._get_full_model_id(model_id, provider)
        
        if full_model_id in self._auto_blacklist:
            del self._auto_blacklist[full_model_id]
            if full_model_id in self._quality_stats:
                self._quality_stats[full_model_id]["blacklisted"] = False
                self._quality_stats[full_model_id]["blacklist_reason"] = None
                # Reset stats to give it a fresh start
                self._quality_stats[full_model_id]["consecutive_failures"] = 0
                self._quality_stats[full_model_id]["http_404_count"] = 0
            self._save_blacklist()
            log_event(f"Removed '{full_model_id}' from auto-blacklist.", "INFO")
            return True
        return False

    def record_quality_score(self, model_id: str, provider: str, score: float) -> None:
        """
        Record a quality score for a model's response.
        
        Args:
            model_id: The model identifier
            provider: The provider (openrouter, horde)
            score: Quality score between 0.0 and 1.0
        """
        if provider not in TRACKED_PROVIDERS:
            return
        
        full_model_id = self._get_full_model_id(model_id, provider)
        stats = self._quality_stats[full_model_id]
        
        # Initialize quality tracking if not present
        if "quality_scores" not in stats:
            stats["quality_scores"] = []
        if "avg_quality_score" not in stats:
            stats["avg_quality_score"] = 0.0
        
        # Keep last 20 scores for rolling average
        stats["quality_scores"].append(score)
        if len(stats["quality_scores"]) > 20:
            stats["quality_scores"] = stats["quality_scores"][-20:]
        
        # Calculate rolling average
        stats["avg_quality_score"] = sum(stats["quality_scores"]) / len(stats["quality_scores"])
        
        log_event(
            f"Quality: Recorded score {score:.2f} for '{full_model_id}' "
            f"(avg: {stats['avg_quality_score']:.2f})",
            "DEBUG"
        )
        
        # Check quality threshold for blacklisting
        if len(stats["quality_scores"]) >= 5 and stats["avg_quality_score"] < QUALITY_SCORE_THRESHOLD:
            if not stats.get("blacklisted"):
                self._blacklist_model(
                    full_model_id, 
                    f"low_quality_responses (avg score: {stats['avg_quality_score']:.2f})"
                )


# --- Standard Benchmark Prompts ---
# These prompts test various capabilities with known correct answers
BENCHMARK_PROMPTS = [
    {
        "id": "math_basic",
        "prompt": "What is 17 * 24? Respond with just the number.",
        "expected_contains": ["408"],
        "category": "math"
    },
    {
        "id": "logic_basic",
        "prompt": "If all cats are animals, and some animals are pets, can we conclude that all cats are pets? Answer yes or no and explain briefly.",
        "expected_contains": ["no", "cannot"],
        "category": "logic"
    },
    {
        "id": "instruction_following",
        "prompt": "List exactly 3 colors. Respond with only the colors, one per line.",
        "validation": lambda x: len([l for l in x.strip().split('\n') if l.strip()]) == 3,
        "category": "instruction"
    },
    {
        "id": "reasoning_basic",
        "prompt": "A farmer has 15 sheep. All but 8 die. How many are left? Just give the number.",
        "expected_contains": ["8"],
        "category": "reasoning"
    },
    {
        "id": "code_basic",
        "prompt": "Write a Python function that returns the sum of two numbers. Just the code, no explanation.",
        "expected_contains": ["def", "return", "+"],
        "category": "code"
    },
]

# Quality score threshold - below this average = blacklist
QUALITY_SCORE_THRESHOLD = 0.4  # 40% quality is minimum acceptable

# LLM Judge prompt template
LLM_JUDGE_PROMPT = """You are an expert evaluator assessing the quality of an LLM response.

BENCHMARK PROMPT:
{prompt}

CANDIDATE MODEL RESPONSE:
{response}

{expected_context}

Evaluate the response on these criteria (0-10 scale each):
1. CORRECTNESS: Is the answer factually accurate and correct?
2. COHERENCE: Is the response well-structured and easy to understand?
3. HELPFULNESS: Does it actually answer the question asked?
4. INSTRUCTION_FOLLOWING: Did it follow any specific format instructions?

Respond ONLY with a JSON object in this exact format:
{{"correctness": N, "coherence": N, "helpfulness": N, "instruction_following": N, "overall": N, "pass": true/false}}

Where N is a number 0-10, and "pass" is true if overall >= 6, false otherwise."""


async def evaluate_model_response(
    response: str,
    benchmark: dict,
    judge_model: str = "gemini-2.5-flash"
) -> dict:
    """
    Use a known-good LLM (Gemini) to evaluate a candidate model's response.
    
    Args:
        response: The candidate model's response to evaluate
        benchmark: The benchmark prompt dict with expected values
        judge_model: The model to use as judge (default: Gemini Flash)
    
    Returns:
        Dict with scores and pass/fail status
    """
    # Quick validation checks first (before calling LLM judge)
    quick_pass = True
    
    # Check expected_contains
    if "expected_contains" in benchmark:
        response_lower = response.lower()
        for expected in benchmark["expected_contains"]:
            if expected.lower() not in response_lower:
                quick_pass = False
                break
    
    # Check custom validation function
    if "validation" in benchmark and callable(benchmark["validation"]):
        try:
            if not benchmark["validation"](response):
                quick_pass = False
        except Exception:
            quick_pass = False
    
    # If quick checks fail badly, skip LLM judge
    if not quick_pass and len(response.strip()) < 10:
        return {
            "correctness": 0, "coherence": 0, "helpfulness": 0,
            "instruction_following": 0, "overall": 0, "pass": False,
            "normalized_score": 0.0
        }
    
    # Build expected context for judge
    expected_context = ""
    if "expected_contains" in benchmark:
        expected_context = f"EXPECTED TO CONTAIN: {', '.join(benchmark['expected_contains'])}"
    
    # Call the LLM judge (Gemini)
    try:
        import google.generativeai as genai
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            log_event("No GEMINI_API_KEY for quality judge, using quick validation only", "WARNING")
            return {
                "correctness": 5 if quick_pass else 2,
                "coherence": 5,
                "helpfulness": 5 if quick_pass else 2,
                "instruction_following": 5 if quick_pass else 2,
                "overall": 5 if quick_pass else 2,
                "pass": quick_pass,
                "normalized_score": 0.5 if quick_pass else 0.2
            }
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(judge_model)
        
        judge_prompt = LLM_JUDGE_PROMPT.format(
            prompt=benchmark["prompt"],
            response=response,
            expected_context=expected_context
        )
        
        result = model.generate_content(judge_prompt)
        response_text = result.text.strip()
        
        # Parse JSON response
        import re
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            scores = json.loads(json_match.group())
            # Normalize to 0-1 scale
            scores["normalized_score"] = scores.get("overall", 5) / 10.0
            return scores
        
    except Exception as e:
        log_event(f"LLM judge evaluation failed: {e}", "WARNING")
    
    # Fallback to quick validation
    return {
        "correctness": 5 if quick_pass else 2,
        "coherence": 5,
        "helpfulness": 5 if quick_pass else 2,
        "instruction_following": 5 if quick_pass else 2,
        "overall": 5 if quick_pass else 2,
        "pass": quick_pass,
        "normalized_score": 0.5 if quick_pass else 0.2
    }


async def run_benchmark_evaluation(model_id: str, provider: str, test_func) -> dict:
    """
    Run all benchmark prompts against a model and return quality report.
    
    Args:
        model_id: The model to test
        provider: The provider (openrouter, horde)
        test_func: Async function to call the model: test_func(prompt) -> response
    
    Returns:
        Dict with benchmark results and overall score
    """
    results = {
        "model_id": model_id,
        "provider": provider,
        "benchmarks": [],
        "passed": 0,
        "failed": 0,
        "avg_score": 0.0
    }
    
    scores = []
    for benchmark in BENCHMARK_PROMPTS:
        try:
            # Call the candidate model
            response = await test_func(benchmark["prompt"])
            
            # Evaluate the response
            evaluation = await evaluate_model_response(response, benchmark)
            
            benchmark_result = {
                "id": benchmark["id"],
                "category": benchmark["category"],
                "passed": evaluation.get("pass", False),
                "score": evaluation.get("normalized_score", 0.0),
                "evaluation": evaluation
            }
            results["benchmarks"].append(benchmark_result)
            scores.append(evaluation.get("normalized_score", 0.0))
            
            if evaluation.get("pass"):
                results["passed"] += 1
            else:
                results["failed"] += 1
                
        except Exception as e:
            log_event(f"Benchmark {benchmark['id']} failed for {model_id}: {e}", "WARNING")
            results["benchmarks"].append({
                "id": benchmark["id"],
                "category": benchmark["category"],
                "passed": False,
                "score": 0.0,
                "error": str(e)
            })
            results["failed"] += 1
            scores.append(0.0)
    
    # Calculate average score
    if scores:
        results["avg_score"] = sum(scores) / len(scores)
    
    # Record quality score with controller
    quality_controller = get_quality_controller()
    quality_controller.record_quality_score(model_id, provider, results["avg_score"])
    
    return results


# --- Singleton Instance ---
_quality_controller: Optional[ModelQualityController] = None


def get_quality_controller() -> ModelQualityController:
    """Get the singleton ModelQualityController instance."""
    global _quality_controller
    if _quality_controller is None:
        _quality_controller = ModelQualityController()
    return _quality_controller

