import copy
import json

def manage_operational_logic(current_logic_state, incoming_directives, priority_assignment, override_conditions):
    """
    Processes incoming directives to update an operational logic state.

    This function takes the current state of a system's logic, a set of incoming directives,
    a way to prioritize them, and conditions for overriding standard processing. It then
    applies the directives to a copy of the state to produce an updated version, without
    affecting the original state.

    Args:
        current_logic_state: A data structure (e.g., dict, list, object) representing the
                             system's existing operational parameters or validation rules.
        incoming_directives: A collection of new instructions or modifications to be integrated.
                             Each directive should be a data structure that can be processed.
        priority_assignment: A function that takes a directive and returns a numerical or
                             comparable value representing its importance (higher values
                             mean higher priority).
        override_conditions: A function that takes a directive and the current state, and
                             returns True if standard validation or processing should be
                             bypassed, False otherwise.

    Returns:
        An updated_logic_state, which is a new data structure reflecting the applied
        directives.
    """
    # Create a deep copy to ensure the original state is not mutated
    updated_logic_state = copy.deepcopy(current_logic_state)

    # Sort directives by priority in descending order (highest priority first)
    sorted_directives = sorted(incoming_directives, key=priority_assignment, reverse=True)

    print("Processing directives...")
    for directive in sorted_directives:
        # Check if any override conditions are met for the current directive
        if override_conditions(directive, updated_logic_state):
            print(f"  - Override condition met for directive: {directive.get('id')}. Bypassing validation.")
            # Apply changes directly. This logic adds new keys or updates existing ones.
            for key, value in directive.get('changes', {}).items():
                updated_logic_state[key] = value
        else:
            print(f"  - Processing directive normally: {directive.get('id')}")
            # In a real-world scenario, validation would occur here.
            # After successful validation, the same application logic is used.
            for key, value in directive.get('changes', {}).items():
                updated_logic_state[key] = value

    return updated_logic_state

if __name__ == "__main__":
    # 1. Define the current_logic_state for a task submission workflow.
    current_logic_state = {
        'submission_enabled': True,
        'deduplication_sensitivity': 'high',
        'allowed_task_types': ['type_a', 'type_b'],
        'max_submission_rate_per_minute': 100,
        'functional_integrity_score': 0.85, # Below the threshold for override
    }

    # 2. Construct incoming_directives to test unified logic.
    incoming_directives = [
        {
            'id': 'directive-002',
            'type': 'PERFORMANCE_TUNING',
            'priority': 5,
            'changes': {
                'deduplication_sensitivity': 'medium',
                'new_setting_standard': 'enabled' # This new key should be added.
            }
        },
        {
            'id': 'directive-001',
            'type': 'CRITICAL_INTEGRITY_FAILSAFE',
            'priority': 10, # Highest priority
            'changes': {
                'submission_enabled': False,
                'new_setting_override': 'critical' # This new key should be added.
            }
        }
    ]

    # 3. Define the priority_assignment mechanism.
    priority_assignment = lambda directive: directive.get('priority', 0)

    # 4. Establish override_conditions.
    def override_conditions(directive, state):
        is_critical_directive = directive.get('type') == 'CRITICAL_INTEGRITY_FAILSAFE'
        is_integrity_compromised = state.get('functional_integrity_score', 1.0) < 0.90
        return is_critical_directive and is_integrity_compromised

    # --- Execution ---
    print("--- Operational Logic Manager Demonstration ---")
    print("\nInitial Logic State:")
    print(json.dumps(current_logic_state, indent=2))
    print("\nIncoming Directives:")
    print(json.dumps(incoming_directives, indent=2))
    print("\n")

    # Invoke the function to get the updated state.
    updated_logic_state = manage_operational_logic(
        current_logic_state,
        incoming_directives,
        priority_assignment,
        override_conditions
    )

    print("\nFinal Updated Logic State (after unified logic):")
    print(json.dumps(updated_logic_state, indent=2))
    print("\n--- End of Demonstration ---")
