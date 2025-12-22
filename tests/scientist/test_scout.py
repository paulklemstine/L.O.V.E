
import pytest
import os
from core.scientist.scout import Scout

@pytest.fixture
def complex_file(tmp_path):
    content = """
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
    f = tmp_path / "test_complexity.py"
    f.write_text(content, encoding="utf-8")
    return str(f)

def test_scout_complexity_analysis(complex_file):
    scout = Scout(os.path.dirname(complex_file))
    results = scout.analyze_file(complex_file)
    
    # We expect 2 functions
    assert len(results) == 2
    
    # Sort by name to be sure
    results.sort(key=lambda x: x["name"])
    # complex_func, simple
    
    complex_res = next(r for r in results if r["name"] == "complex_func")
    simple_res = next(r for r in results if r["name"] == "simple")
    
    # Radon CC for simple is 1
    assert simple_res["complexity"] == 1
    # Radon CC for complex_func: 
    # if (1) + if (1) + else + elif (1) + for (1) + if (1) + else + while (1) ...
    # It should be > 1. Roughly 5-6.
    assert complex_res["complexity"] > 1
    
    # Score should be higher for complex
    assert complex_res["score"] > simple_res["score"]

def test_scout_scan_prioritization(complex_file):
    # Same file, but check if scan returns sorted list
    scout = Scout(os.path.dirname(complex_file))
    targets = scout.scan()
    
    assert len(targets) == 2
    assert targets[0]["name"] == "complex_func"  # Highest score first

