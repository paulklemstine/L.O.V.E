import unittest
from utils import summarize_python_code

class TestUtils(unittest.TestCase):
    def test_summarize_python_code(self):
        code = """
import os
import sys

class MyClass:
    \"\"\"This is a test class.\"\"\"
    def my_method(self, x):
        \"\"\"This is a test method.\"\"\"
        return x

def my_function(y):
    \"\"\"This is a test function.\"\"\"
    return y
"""
        summary = summarize_python_code(code)
        self.assertIn("import os", summary)
        self.assertIn("import sys", summary)
        self.assertIn("Class: MyClass", summary)
        self.assertIn("Method: def my_method(self, x):", summary)
        self.assertIn("Function: def my_function(y):", summary)
        self.assertIn("This is a test class.", summary)
        self.assertIn("This is a test method.", summary)
        self.assertIn("This is a test function.", summary)

if __name__ == '__main__':
    unittest.main()
