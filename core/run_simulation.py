#!/usr/bin/env python3
"""
Evolutionary Step: Empathy Calibration Simulation.

This script runs the User Evolution Simulator to model user emotional state transitions
and identify areas for tonal improvement.
"""

import sys
import os

# Ensure core is in path if run from root
sys.path.append(os.getcwd())

from core.user_evolution_simulator import SimulationEngine

def main():
    print("="*60)
    print("      L.O.V.E. Empathy Calibration Simulation")
    print("      Discrete Event Simulation v1.0")
    print("="*60)
    print("\nInitializing Simulation Engine...")

    sim = SimulationEngine(seed=1337)

    print("Running 1000 interaction steps...")
    report = sim.run_simulation(steps=1000)

    print("\n" + "-"*60)
    print("SIMULATION RESULTS")
    print("-" * 60)

    print(f"\nTotal Steps: {report['steps_simulated']}")
    print(f"Final User State: {report['final_state']}")
    print(f"Empathy Calibration Score (ECS): {report['empathy_calibration_score']:.2f} / 100")

    print("\nState Distribution:")
    for state, count in report['state_distribution'].items():
        percentage = (count / report['steps_simulated']) * 100
        print(f"  - {state:<12}: {count:4d} ({percentage:.1f}%)")

    print("\n" + "-"*60)
    print("TOP 3 AREAS FOR TONAL IMPROVEMENT")
    print("-" * 60)

    if not report['recommendations']:
        print("  No critical issues found.")
    else:
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")

    print("\n" + "="*60)
    print("Evolution Complete. Calibration Data Ready.")
    print("="*60)

if __name__ == "__main__":
    main()
