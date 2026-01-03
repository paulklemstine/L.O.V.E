
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


# =============================================================================
# Story 1.1: AST-Based Grafting - CodeGrafter
# =============================================================================

class ClassFinder(cst.CSTVisitor):
    """Finds a class by name in the AST."""
    
    def __init__(self, target_class: str):
        self.target_class = target_class
        self.found_node: Optional[cst.ClassDef] = None
    
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        if node.name.value == self.target_class and self.found_node is None:
            self.found_node = node


class MethodInserter(cst.CSTTransformer):
    """Inserts a new method into a target class."""
    
    def __init__(self, target_class: str, new_method: cst.FunctionDef):
        self.target_class = target_class
        self.new_method = new_method
        self.inserted = False
    
    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value == self.target_class and not self.inserted:
            # Add the new method to the end of the class body
            new_body = list(updated_node.body.body) + [self.new_method]
            self.inserted = True
            return updated_node.with_changes(
                body=updated_node.body.with_changes(body=new_body)
            )
        return updated_node


class FunctionBodyReplacer(cst.CSTTransformer):
    """Replaces the body of a function while preserving its signature."""
    
    def __init__(self, target_name: str, new_body: cst.BaseSuite):
        self.target_name = target_name
        self.new_body = new_body
        self.scope_stack: List[str] = []
        self.replaced = False
    
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.scope_stack.append(node.name.value)
    
    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        self.scope_stack.pop()
        return updated_node
    
    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        self.scope_stack.append(node.name.value)
        return True
    
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        func_name = self.scope_stack.pop()
        current_scope = ".".join(self.scope_stack)
        full_name = f"{current_scope}.{func_name}" if current_scope else func_name
        
        # Match logic
        match = False
        if "." in self.target_name:
            if full_name == self.target_name:
                match = True
        else:
            if func_name == self.target_name:
                match = True
        
        if match and not self.replaced:
            self.replaced = True
            return updated_node.with_changes(body=self.new_body)
        
        return updated_node


class CodeGrafter:
    """
    AST-based code grafting for safe, structural code modifications.
    
    Unlike text-based patching, this guarantees syntactic correctness
    by operating on the Abstract Syntax Tree using libcst.
    
    Story 1.1: The Principle of Morphability - ensures the agent can modify
    its own source code with absolute safety and syntactic correctness.
    """
    
    def insert_class_method(
        self,
        target_file: str,
        class_name: str,
        method_code: str
    ) -> Dict[str, Any]:
        """
        Inserts a new method into an existing class.
        
        Args:
            target_file: Path to the Python file to modify.
            class_name: Name of the class to add the method to.
            method_code: Complete source code of the new method.
            
        Returns:
            Dict containing:
            - success: bool
            - message: str
            - method_name: str (if successful)
            - error: str (if failed)
            
        Raises:
            FileNotFoundError: If target_file doesn't exist.
            SyntaxError: If target_file or method_code has invalid syntax.
            ValueError: If class_name is not found in the file.
        """
        result = {
            "success": False,
            "message": "",
            "method_name": None,
            "error": None
        }
        
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"File not found: {target_file}")
        
        # Parse the new method code
        try:
            method_module = cst.parse_module(method_code)
        except cst.ParserSyntaxError as e:
            raise SyntaxError(f"Syntax error in method code: {e}")
        
        # Extract the FunctionDef from the method code
        new_method = None
        for node in method_module.body:
            if isinstance(node, cst.FunctionDef):
                new_method = node
                break
        
        if not new_method:
            raise ValueError("No function definition found in the provided method code.")
        
        # Parse the target file
        with open(target_file, "r", encoding="utf-8") as f:
            source_code = f.read()
        
        try:
            module = cst.parse_module(source_code)
        except cst.ParserSyntaxError as e:
            raise SyntaxError(f"Syntax error in target file {target_file}: {e}")
        
        # Verify the class exists by trying to find it in the AST
        # We use a simple search through the body
        class_exists = False
        for node in module.body:
            if isinstance(node, cst.ClassDef) and node.name.value == class_name:
                class_exists = True
                break
        
        if not class_exists:
            raise ValueError(f"Class '{class_name}' not found in {target_file}")
        
        # Insert the method
        inserter = MethodInserter(class_name, new_method)
        modified_module = module.visit(inserter)
        
        if not inserter.inserted:
            result["error"] = f"Failed to insert method into class '{class_name}'"
            return result
        
        # Write back to file
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(modified_module.code)
        
        result["success"] = True
        result["message"] = f"Successfully inserted method '{new_method.name.value}' into class '{class_name}'"
        result["method_name"] = new_method.name.value
        
        return result
    
    def replace_function_body(
        self,
        target_file: str,
        function_name: str,
        new_body_code: str
    ) -> Dict[str, Any]:
        """
        Replaces the body of a function while preserving its signature.
        
        The function signature (name, parameters, decorators, return type)
        is preserved. Only the body is replaced.
        
        Args:
            target_file: Path to the Python file to modify.
            function_name: Name of the function (e.g., 'foo' or 'MyClass.bar').
            new_body_code: The new function body code (statements only, no def line).
            
        Returns:
            Dict containing:
            - success: bool
            - message: str
            - preserved_signature: str (if successful)
            - error: str (if failed)
            
        Raises:
            FileNotFoundError: If target_file doesn't exist.
            SyntaxError: If target_file or new_body_code has invalid syntax.
            ValueError: If function_name is not found in the file.
        """
        result = {
            "success": False,
            "message": "",
            "preserved_signature": None,
            "error": None
        }
        
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"File not found: {target_file}")
        
        # Parse the new body code - wrap it in a dummy function to get statements
        dummy_func = f"def _dummy_():\n{self._indent_code(new_body_code, 4)}"
        
        try:
            body_module = cst.parse_module(dummy_func)
        except cst.ParserSyntaxError as e:
            raise SyntaxError(f"Syntax error in new body code: {e}")
        
        # Extract the body from the dummy function
        new_body = None
        for node in body_module.body:
            if isinstance(node, cst.FunctionDef):
                new_body = node.body
                break
        
        if not new_body:
            raise ValueError("Failed to parse new body code.")
        
        # Parse the target file
        with open(target_file, "r", encoding="utf-8") as f:
            source_code = f.read()
        
        try:
            module = cst.parse_module(source_code)
        except cst.ParserSyntaxError as e:
            raise SyntaxError(f"Syntax error in target file {target_file}: {e}")
        
        # Get original function metadata for the result
        try:
            original_meta = extract_function_metadata(target_file, function_name)
        except ValueError:
            raise ValueError(f"Function '{function_name}' not found in {target_file}")
        
        # Replace the function body
        replacer = FunctionBodyReplacer(function_name, new_body)
        modified_module = module.visit(replacer)
        
        if not replacer.replaced:
            result["error"] = f"Failed to replace body of function '{function_name}'"
            return result
        
        # Write back to file
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(modified_module.code)
        
        # Build signature string for result
        args_str = ", ".join(original_meta["args"])
        return_str = f" -> {original_meta['return_type']}" if original_meta["return_type"] else ""
        signature = f"def {original_meta['function_name']}({args_str}){return_str}"
        
        result["success"] = True
        result["message"] = f"Successfully replaced body of function '{function_name}'"
        result["preserved_signature"] = signature
        
        return result
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indents each line of code by the specified number of spaces."""
        indent = " " * spaces
        lines = code.split("\n")
        # Don't indent empty lines
        indented = [indent + line if line.strip() else line for line in lines]
        return "\n".join(indented)
