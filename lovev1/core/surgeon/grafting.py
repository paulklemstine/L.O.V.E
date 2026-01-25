
import libcst as cst
import os
from typing import Optional, List, Union

class FunctionReplacer(cst.CSTTransformer):
    def __init__(self, target_name: str, new_node: cst.FunctionDef):
        self.target_name = target_name
        self.new_node = new_node
        self.scope_stack: List[str] = []
        self.replaced = False

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.scope_stack.append(node.name.value)
        # print(f"DEBUG: Entered Class {node.name.value}, stack: {self.scope_stack}")

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        self.scope_stack.pop()
        # print(f"DEBUG: Left Class {original_node.name.value}, stack: {self.scope_stack}")
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        # 1. Calculate name relative to current scope
        current_scope = ".".join(self.scope_stack)
        func_name = node.name.value
        # full_name = f"{current_scope}.{func_name}" if current_scope else func_name
        
        # print(f"DEBUG: Visiting Function {full_name}, stack before push: {self.scope_stack}")

        self.scope_stack.append(func_name)
        return True # Visit children

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> Union[cst.FunctionDef, cst.RemovalSentinel]:
        func_name = self.scope_stack.pop() # Pop self
        
        current_scope = ".".join(self.scope_stack)
        full_name = f"{current_scope}.{func_name}" if current_scope else func_name
        
        # print(f"DEBUG: Leaving Function {full_name}, target: {self.target_name}, replaced: {self.replaced}")

        match = False
        if "." in self.target_name:
            if full_name == self.target_name:
                match = True
        else:
            if func_name == self.target_name:
                match = True
        
        if match and not self.replaced:
            # print(f"DEBUG: MATCH FOUND for {full_name}")
            self.replaced = True
            return self.new_node
            
        return updated_node


def graft_function(file_path: str, target_name: str, new_code: str) -> None:
    """
    Replaces a function in a file with new code safe, preserving formatting.
    
    Args:
        file_path: Path to the python file.
        target_name: Name of the function to replace (e.g. 'foo' or 'MyClass.bar').
        new_code: Complete source code of the new function.
        
    Raises:
        FileNotFoundError
        SyntaxError: If file or new_code is invalid.
        ValueError: If target function is not found.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Parse new code to ensure validity and get the node
    try:
        new_module = cst.parse_module(new_code)
    except cst.ParserSyntaxError as e:
        raise SyntaxError(f"Syntax error in new code: {e}")

    # Extract the FunctionDef from the new code
    target_leaf_name = target_name.split(".")[-1]
    
    new_func_node = None
    for node in new_module.body:
        if isinstance(node, cst.FunctionDef):
            if node.name.value == target_leaf_name:
                new_func_node = node
                break
    
    if not new_func_node:
        # Fallback: take the first function found
        for node in new_module.body:
            if isinstance(node, cst.FunctionDef):
                new_func_node = node
                break
                
    if not new_func_node:
        raise ValueError("No function definition found in the provided new code.")

    # 2. Parse original file
    with open(file_path, "r", encoding="utf-8") as f:
        original_source = f.read()

    try:
        original_module = cst.parse_module(original_source)
    except cst.ParserSyntaxError as e:
        raise SyntaxError(f"Syntax error in target file {file_path}: {e}")

    # 3. Perform Replacement
    replacer = FunctionReplacer(target_name, new_func_node)
    
    modified_module = original_module.visit(replacer)

    if not replacer.replaced:
        raise ValueError(f"Target function '{target_name}' not found in {file_path}")

    # 4. Write back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(modified_module.code)
