
from radon.complexity import cc_visit
code = """
def simple():
    return 1

def complex_func(a, b):
    if a > b:
        if a > 10:
            return a
        else:
            return b
    elif b > a:
        for i in range(b):
            if i % 2 == 0:
                print(i)
    else:
        while True:
            break
    return 0
"""
try:
    blocks = cc_visit(code)
    print(f"Blocks found: {len(blocks)}")
    for b in blocks:
        print(f"Block: {b.name}, Type: {getattr(b, 'type', 'N/A')}, Complexity: {b.complexity}")
except Exception as e:
    print(f"Error: {e}")
