
import libcst as cst
from libcst.metadata import PositionProvider, MetadataWrapper
from typing import Optional, Dict, Any, List, Union
import os

class FunctionFinder(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, target_name: str):
        self.target_name = target_name
        self.found_node: Optional[cst.FunctionDef] = None
        self.scope_stack: List[str] = []
        self.found_scope: str = ""

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.scope_stack.append(node.name.value)

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        current_scope = ".".join(self.scope_stack)
        func_name = node.name.value
        full_name = f"{current_scope}.{func_name}" if current_scope else func_name

        # Match logic:
        # If target_name has dots, exact match required.
        # If target_name has no dots, match the function name (leaf).
        
        match = False
        if "." in self.target_name:
            if full_name == self.target_name:
                match = True
        else:
            if func_name == self.target_name:
                match = True
        
        if match and self.found_node is None:
             self.found_node = node
             self.found_scope = current_scope
        
        self.scope_stack.append(func_name)

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.scope_stack.pop()

def extract_function_metadata(file_path: str, function_name: str) -> Dict[str, Any]:
    """
    Parses a file and extracts metadata for a specific function/method.
    
    Args:
        file_path: Path to the python file.
        function_name: Name of the function (e.g., 'my_func' or 'MyClass.my_method').
        
    Returns:
        Dict containing:
        - source: str
        - start_line: int
        - end_line: int
        - args: List[str]
        - return_annotation: str (or None)
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        SyntaxError: If file is invalid python.
        ValueError: If function not found.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    try:
        module = cst.parse_module(source_code)
    except cst.ParserSyntaxError as e:
        raise SyntaxError(f"Syntax error in structure of {file_path}: {e}")

    wrapper = MetadataWrapper(module)
    visitor = FunctionFinder(function_name)
    wrapper.visit(visitor)

    if not visitor.found_node:
        raise ValueError(f"Function '{function_name}' not found in {file_path}")

    node = visitor.found_node
    
    # Extract position
    pos = wrapper.resolve(PositionProvider)[node]
    start_line = pos.start.line
    end_line = pos.end.line
    
    # Extract args
    args = []
    for param in node.params.params:
        arg_name = param.name.value
        annotation = ""
        if param.annotation:
            annotation = f": {cst.Module([]).code_for_node(param.annotation.annotation)}"
        args.append(f"{arg_name}{annotation}")
        
    # Extract return annotation
    return_type = None
    if node.returns:
         return_type = cst.Module([]).code_for_node(node.returns.annotation)

    # Extract source code exactly as is in the file (preserving whitespace/comments for that block)
    # Ideally we use the module code and slice it, or use node.code if it preserves original formatting.
    # LibCST node.code usually preserves formatting if parsed from source?
    # Actually LibCST regenerates code. To get exact source, we should use line numbers and the original source string.
    # However, LibCST `code_for_node` should return the source implementation from the tree.
    # BUT, if we want *exact* original bytes including weird spacing that LibCST might normalize?
    # LibCST is designed to preserve everything, so `module.code_for_node(node)` should be good.
    
    node_source = module.code_for_node(node)

    return {
        "function_name": node.name.value,
        "source": node_source,
        "start_line": start_line,
        "end_line": end_line,
        "args": args,
        "return_type": return_type
    }
