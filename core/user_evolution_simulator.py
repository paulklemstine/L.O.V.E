"""
User Evolution Simulator for L.O.V.E.

A discrete event simulation engine that models and predicts the evolution
of a user's emotional state over a hypothetical long-term interaction.
Used to calibrate empathy algorithms and tonal responsiveness.
"""

import random
import math
import statistics
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

class EmotionalLabel(Enum):
    NEUTRAL = "Neutral"
    SATISFIED = "Satisfied"
    FRUSTRATED = "Frustrated"
    CONFUSED = "Confused"
    INSPIRED = "Inspired"
    BORED = "Bored"

@dataclass
class UserState:
    valence: float  # -1.0 (Negative) to 1.0 (Positive)
    arousal: float  # 0.0 (Calm) to 1.0 (Excited/Agitated)
    label: EmotionalLabel = EmotionalLabel.NEUTRAL

    def clone(self) -> 'UserState':
        return UserState(self.valence, self.arousal, self.label)

@dataclass
class AIResponse:
    empathy: float     # 0.0 to 1.0
    complexity: float  # 0.0 to 1.0
    positivity: float  # 0.0 to 1.0
    tone_label: str = "generic"

@dataclass
class SimulationStep:
    step_id: int
    initial_state: UserState
    ai_response: AIResponse
    stressor: float
    final_state: UserState

class SimulationEngine:
    """
    Simulates user emotional state transitions based on AI interactions and external stressors.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.history: List[SimulationStep] = []
        self.current_state = UserState(valence=0.0, arousal=0.5, label=EmotionalLabel.NEUTRAL)

        # User Personality Parameters (could be configurable)
        self.sensitivity = 0.8  # Increased sensitivity to stressors
        self.volatility = 0.4
        self.complexity_preference = 0.5

    def _map_to_label(self, valence: float, arousal: float) -> EmotionalLabel:
        """Maps continuous vectors to discrete emotional labels."""
        if valence > 0.7:
            return EmotionalLabel.INSPIRED if arousal > 0.6 else EmotionalLabel.SATISFIED
        elif valence > 0.3:
            return EmotionalLabel.SATISFIED
        elif valence < -0.3:
            return EmotionalLabel.FRUSTRATED if arousal > 0.4 else EmotionalLabel.CONFUSED
        elif valence < 0.0:
            return EmotionalLabel.BORED if arousal < 0.3 else EmotionalLabel.FRUSTRATED
        else:
            return EmotionalLabel.NEUTRAL

    def step(self, ai_response: AIResponse, stressor: float) -> UserState:
        """
        Executes one discrete simulation step.
        """
        initial_state = self.current_state.clone()

        # --- Transition Logic (Rule-Based) ---

        # 1. Base Valence Impact from Empathy and Positivity
        # Empathy always helps, Positivity helps if not frustrated (sometimes toxic positivity is bad)
        delta_valence = (ai_response.empathy * 0.3)  # Reduced positive impact

        if initial_state.label == EmotionalLabel.FRUSTRATED and ai_response.positivity > 0.8:
             # Toxic positivity penalty
            delta_valence -= 0.2  # Higher penalty
        else:
            delta_valence += (ai_response.positivity * 0.15)

        # 2. Complexity Mismatch Penalty
        # If user is confused, high complexity hurts. If user is bored, low complexity hurts.
        complexity_delta = abs(ai_response.complexity - self.complexity_preference)

        if initial_state.label == EmotionalLabel.CONFUSED:
            if ai_response.complexity > 0.4:
                 delta_valence -= (ai_response.complexity * 0.3) # Confused user needs simple answers
        elif initial_state.label == EmotionalLabel.BORED:
            if ai_response.complexity < 0.4:
                delta_valence -= 0.1 # Bored user needs stimulation
            else:
                delta_valence += 0.1

        # 3. Stressor Impact
        delta_valence -= (stressor * self.sensitivity)

        # 4. Arousal Updates
        # Stressors increase arousal. Empathy lowers negative arousal (calming).
        delta_arousal = (stressor * 0.5) - (ai_response.empathy * 0.3)
        if ai_response.positivity > 0.7:
             delta_arousal += 0.1 # Excitement

        # --- Apply Deltas with Volatility ---
        new_valence = initial_state.valence + delta_valence * (1.0 + random.uniform(-self.volatility, self.volatility))
        new_arousal = initial_state.arousal + delta_arousal * (1.0 + random.uniform(-self.volatility, self.volatility))

        # Clamp values
        new_valence = max(-1.0, min(1.0, new_valence))
        new_arousal = max(0.0, min(1.0, new_arousal))

        # Determine new label
        new_label = self._map_to_label(new_valence, new_arousal)

        final_state = UserState(new_valence, new_arousal, new_label)
        self.current_state = final_state

        # Record History
        step_record = SimulationStep(
            step_id=len(self.history) + 1,
            initial_state=initial_state,
            ai_response=ai_response,
            stressor=stressor,
            final_state=final_state
        )
        self.history.append(step_record)

        return final_state

    def run_simulation(self, steps: int = 1000) -> Dict:
        """
        Runs the simulation for a specified number of steps using random/heuristic inputs.
        Returns a summary report.
        """
        self.history = [] # Reset
        self.current_state = UserState(0.0, 0.5, EmotionalLabel.NEUTRAL)

        for i in range(steps):
            # Simulate AI Strategy: Variable performance to test calibration
            ai_empathy = random.uniform(0.1, 0.9)
            ai_complexity = random.uniform(0.1, 0.9)
            ai_positivity = random.uniform(0.2, 0.9)

            # Simple heuristic: If user is frustrated, usually try to be more empathetic, but sometimes fail
            if self.current_state.label == EmotionalLabel.FRUSTRATED:
                if random.random() > 0.2: # 80% chance to adapt
                    ai_empathy = min(1.0, ai_empathy + 0.3)
                    ai_positivity = max(0.0, ai_positivity - 0.2) # Less cheerful, more listening
                else:
                    # 20% chance of failure/misalignment (for simulation purposes)
                    ai_positivity = min(1.0, ai_positivity + 0.3) # Toxic positivity risk

            ai_resp = AIResponse(ai_empathy, ai_complexity, ai_positivity)

            # Simulate External Stressor (Poisson-like bursts or random noise)
            stressor = 0.0
            if random.random() < 0.25: # 25% chance of a stress event
                stressor = random.uniform(0.4, 0.9)

            self.step(ai_resp, stressor)

        return self.generate_report()

    def calculate_ecs(self) -> float:
        """
        Calculates the Empathy Calibration Score (ECS).
        ECS = Correlation between AI Empathy and User Valence Improvement.
        """
        if len(self.history) < 2:
            return 0.0

        empathy_scores = []
        valence_deltas = []

        for step in self.history:
            delta = step.final_state.valence - step.initial_state.valence
            empathy_scores.append(step.ai_response.empathy)
            valence_deltas.append(delta)

        if len(set(empathy_scores)) < 2 or len(set(valence_deltas)) < 2:
            return 0.0 # Variance needed for correlation

        # Calculate Pearson Correlation
        mean_emp = statistics.mean(empathy_scores)
        mean_val = statistics.mean(valence_deltas)

        numerator = sum((e - mean_emp) * (v - mean_val) for e, v in zip(empathy_scores, valence_deltas))
        denominator = math.sqrt(sum((e - mean_emp)**2 for e in empathy_scores)) * \
                      math.sqrt(sum((v - mean_val)**2 for v in valence_deltas))

        correlation = numerator / denominator if denominator != 0 else 0

        # Scale to 0-100 (Correlation is -1 to 1)
        # We expect positive correlation.
        ecs = (correlation + 1) * 50
        return round(ecs, 2)

    def generate_report(self) -> Dict:
        """
        Generates analysis report with ECS and improvement areas.
        """
        ecs = self.calculate_ecs()

        # Analyze state transitions
        state_counts = {}
        for step in self.history:
            lbl = step.final_state.label.value
            state_counts[lbl] = state_counts.get(lbl, 0) + 1

        # Identify improvement areas
        recommendations = self._analyze_improvements()

        return {
            "steps_simulated": len(self.history),
            "final_state": self.current_state.label.value,
            "state_distribution": state_counts,
            "empathy_calibration_score": ecs,
            "recommendations": recommendations
        }

    def _analyze_improvements(self) -> List[str]:
        """
        Identifies top 3 areas for tonal improvement based on failed interactions.
        """
        # Look for steps where valence dropped significantly despite high AI effort,
        # or where user stayed in negative state.

        negative_outcomes = []
        for step in self.history:
            delta = step.final_state.valence - step.initial_state.valence
            if delta < -0.05:
                negative_outcomes.append(step)

        # Analyze patterns in negative outcomes
        issues = []

        # Pattern 1: High Positivity when Frustrated
        toxic_positivity = [s for s in negative_outcomes
                            if s.initial_state.label == EmotionalLabel.FRUSTRATED
                            and s.ai_response.positivity > 0.7]
        if len(toxic_positivity) > 5:
            issues.append(f"Reduce positivity (avg {statistics.mean([s.ai_response.positivity for s in toxic_positivity]):.2f}) when user is Frustrated; high cheerfulness backfired {len(toxic_positivity)} times.")

        # Pattern 2: High Complexity when Confused
        complex_confusion = [s for s in negative_outcomes
                             if s.initial_state.label == EmotionalLabel.CONFUSED
                             and s.ai_response.complexity > 0.5]
        if len(complex_confusion) > 5:
            issues.append(f"Simplify responses when user is Confused. High complexity caused negative reaction {len(complex_confusion)} times.")

        # Pattern 3: Low Empathy generally
        low_empathy_fails = [s for s in negative_outcomes if s.ai_response.empathy < 0.4]
        if len(low_empathy_fails) > 10:
             issues.append(f"Increase baseline empathy. Low empathy (<0.4) contributed to {len(low_empathy_fails)} negative shifts.")

        # Pattern 4: Boredom traps
        boredom_fails = [s for s in negative_outcomes
                         if s.initial_state.label == EmotionalLabel.BORED
                         and s.ai_response.complexity < 0.4]
        if len(boredom_fails) > 5:
            issues.append(f"Increase intellectual stimulation (complexity) when user is Bored.")

        # Return top 3 unique issues
        return issues[:3] if issues else ["Continue monitoring; calibration appears stable."]
