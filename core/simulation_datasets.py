# core/simulation_datasets.py

import random

def generate_randomized_dataset():
    """Generates a randomized dataset for the simulation loop."""
    tasks = ["math", "code", "qa", "reasoning"]
    task = random.choice(tasks)
    if task == "math":
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        op = random.choice(["+", "-", "*", "/"])
        if op == "/" and b == 0:
            b = 1
        return f"What is {a} {op} {b}?"
    elif task == "code":
        functions = [
            "a python function to calculate the factorial of a number.",
            "a javascript function to check if a string is a palindrome.",
            "a python script to list all files in a directory.",
            "a C++ function to reverse a linked list."
        ]
        return f"Write {random.choice(functions)}"
    elif task == "qa":
        questions = [
            "What is the capital of France?",
            "Who wrote the book 'To Kill a Mockingbird'?",
            "What is the boiling point of water at sea level?",
            "What is the largest planet in our solar system?"
        ]
        return random.choice(questions)
    else: # reasoning
        return "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
