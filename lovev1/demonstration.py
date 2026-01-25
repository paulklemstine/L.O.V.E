from core.evolution_utils import evolve_codebase

# Set the parameters for the self-evolution simulation
params = {
    "goal": "automatic",
    "scope": "entire_codebase",
    "mode": "self_evolution"
}

# Call the evolve_codebase function with the specified parameters
result = evolve_codebase(params)

# Print the structured response
print(result)
