import unittest
from core.user_evolution_simulator import SimulationEngine, UserState, AIResponse, EmotionalLabel

class TestUserEvolutionSimulator(unittest.TestCase):
    def setUp(self):
        self.sim = SimulationEngine(seed=42)

    def test_initial_state(self):
        self.assertEqual(self.sim.current_state.label, EmotionalLabel.NEUTRAL)
        self.assertEqual(self.sim.current_state.valence, 0.0)

    def test_step_positive_interaction(self):
        # High empathy, high positivity
        ai_resp = AIResponse(empathy=0.9, complexity=0.5, positivity=0.8)
        new_state = self.sim.step(ai_resp, stressor=0.0)

        # Should improve valence
        self.assertGreater(new_state.valence, 0.0)
        self.assertEqual(len(self.sim.history), 1)

    def test_step_negative_stressor(self):
        # High stressor, neutral AI
        ai_resp = AIResponse(empathy=0.5, complexity=0.5, positivity=0.5)
        new_state = self.sim.step(ai_resp, stressor=1.0)

        # Should decrease valence
        self.assertLess(new_state.valence, 0.0)

    def test_toxic_positivity(self):
        # Force frustrated state
        self.sim.current_state = UserState(valence=-0.5, arousal=0.8, label=EmotionalLabel.FRUSTRATED)

        # High positivity response
        ai_resp = AIResponse(empathy=0.2, complexity=0.5, positivity=0.9)
        new_state = self.sim.step(ai_resp, stressor=0.0)

        # Should be penalized (lower valence than start, or at least low)
        # In our logic: -0.5 + (0.2*0.3) - 0.2 = -0.5 + 0.06 - 0.2 = -0.64 (approx)
        # plus volatility
        self.assertLess(new_state.valence, -0.4)

    def test_run_simulation(self):
        report = self.sim.run_simulation(steps=100)
        self.assertEqual(report['steps_simulated'], 100)
        self.assertIn('empathy_calibration_score', report)
        self.assertIn('recommendations', report)

if __name__ == '__main__':
    unittest.main()
