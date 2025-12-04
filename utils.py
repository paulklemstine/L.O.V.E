import subprocess
import os
import sys
import platform
import ast
import shutil
import logging
import re


def get_network_interfaces(autopilot_mode=False):
    """
    Retrieves detailed information about all network interfaces.
    Returns a tuple of (details_dict, error_string).
    """
    import netifaces
    interfaces = {}
    try:
        for iface in netifaces.interfaces():
            details = netifaces.ifaddresses(iface)
            interfaces[iface] = {
                "mac": details.get(netifaces.AF_LINK, [{}])[0].get('addr', 'N/A'),
                "ipv4": details.get(netifaces.AF_INET, [{}])[0],
                "ipv6": details.get(netifaces.AF_INET6, [{}])[0],
            }
        # For autopilot, return a simple string summary
        if autopilot_mode:
            summary_lines = []
            for iface, data in interfaces.items():
                if data['ipv4'].get('addr'):
                    summary_lines.append(f"{iface}: {data['ipv4']['addr']}")
            return interfaces, ", ".join(summary_lines)
        return interfaces, None
    except Exception as e:
        logging.error(f"Could not get network interface details: {e}")
        return None, f"Error getting network interface details: {e}"


def get_git_repo_info():
    """Retrieves the GitHub repository owner, name, and current commit hash."""
    try:
        # Get current commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        commit_hash = hash_result.stdout.strip()

        # Get all remotes
        remotes_result = subprocess.run(
            ["git", "remote"],
            capture_output=True,
            text=True,
            check=True
        )
        remotes = remotes_result.stdout.strip().splitlines()

        if not remotes:
            return None

        # Try each remote until we find a valid GitHub URL
        for remote_name in remotes:
            url_result = subprocess.run(
                ["git", "config", "--get", f"remote.{remote_name}.url"],
                capture_output=True,
                text=True,
                check=True
            )
            url = url_result.stdout.strip()

            # Extract owner and repo name
            if "github.com" in url:
                owner = None
                repo = None
                if url.startswith("git@"):
                    # SSH format: git@github.com:owner/repo.git
                    parts = url.split(":")[1].split("/")
                    owner = parts[0]
                    repo = parts[1].replace(".git", "")
                elif url.startswith("https://"):
                    # HTTPS format: https://github.com/owner/repo.git
                    parts = url.split("/")
                    owner = parts[-2]
                    repo = parts[-1].replace(".git", "")
                
                if owner and repo:
                    return {
                        "owner": owner,
                        "repo": repo,
                        "hash": commit_hash
                    }

        return None  # No GitHub URL found in any remote

    except (subprocess.CalledProcessError, IndexError):
        return None


def list_directory(path="."):
    """
    Lists the contents of a directory.
    Returns a tuple of (list_of_contents, error_string).
    """
    if not os.path.isdir(path):
        return None, f"Error: Directory not found at '{path}'"
    try:
        # Use a more informative listing format, similar to 'ls -la'
        result = subprocess.run(
            ["ls", "-la", path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        return None, f"Error listing directory '{path}': {e.stderr}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

def summarize_python_code(code: str, mode: str = "full", max_tokens: int = None) -> str:
    """
    Summarizes Python code using AST to extract imports, class and function
    definitions. Supports multiple modes for different levels of detail.
    
    Args:
        code: Python source code to summarize
        mode: Summary mode - "full" (default), "signatures", or "minimal"
            - "full": Signatures + full docstrings
            - "signatures": Signatures only, no docstrings
            - "minimal": Signatures only, no imports, ultra-compact
        max_tokens: Optional token budget (rough estimation: 1 token ≈ 4 chars)
    
    Returns:
        Formatted summary string
    """
    summary = []
    try:
        tree = ast.parse(code)

        # Extract imports (skip in minimal mode)
        if mode != "minimal":
            imports = []
            for node in tree.body:
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    imports.append(ast.unparse(node))

            if imports:
                summary.append("Imports:")
                for imp in imports:
                    summary.append(f"  - {imp}")
                summary.append("") # Add a blank line for spacing

        # Extract functions and classes
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Get docstring only in full mode
                docstring = None
                if mode == "full":
                    docstring = ast.get_docstring(node, clean=True)
                
                # Reconstruct signature without the body
                func_node_copy = ast.FunctionDef(
                    name=node.name, 
                    args=node.args, 
                    returns=node.returns, 
                    decorator_list=node.decorator_list, 
                    body=[]
                )
                ast.copy_location(func_node_copy, node)
                signature = ast.unparse(func_node_copy).strip()

                if mode == "minimal":
                    # Ultra-compact: just the signature
                    summary.append(signature)
                else:
                    summary.append(f"Function: {signature}")
                    if docstring:
                        indented_docstring = "\n".join([f"    {line}" for line in docstring.splitlines()])
                        summary.append(f"  - Docstring:\n{indented_docstring}")

            elif isinstance(node, ast.ClassDef):
                # Get docstring only in full mode
                docstring = None
                if mode == "full":
                    docstring = ast.get_docstring(node, clean=True)
                
                if mode == "minimal":
                    summary.append(f"class {node.name}:")
                else:
                    summary.append(f"Class: {node.name}")
                    if docstring:
                        indented_docstring = "\n".join([f"    {line}" for line in docstring.splitlines()])
                        summary.append(f"  - Docstring:\n{indented_docstring}")

                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        # Get docstring only in full mode
                        method_docstring = None
                        if mode == "full":
                            method_docstring = ast.get_docstring(method, clean=True)
                        
                        # Reconstruct signature without the body
                        method_node_copy = ast.FunctionDef(
                            name=method.name, 
                            args=method.args, 
                            returns=method.returns, 
                            decorator_list=method.decorator_list, 
                            body=[]
                        )
                        ast.copy_location(method_node_copy, method)
                        method_signature = ast.unparse(method_node_copy).strip()

                        if mode == "minimal":
                            summary.append(f"  {method_signature}")
                        else:
                            summary.append(f"  - Method: {method_signature}")
                            if method_docstring:
                                indented_method_docstring = "\n".join([f"      {line}" for line in method_docstring.splitlines()])
                                summary.append(f"    - Docstring:\n{indented_method_docstring}")
        
        # Apply token budget if specified
        summary_text = "\n".join(summary)
        if max_tokens:
            max_chars = max_tokens * 4  # Rough estimation
            if len(summary_text) > max_chars:
                # Truncate with indicator
                summary_text = summary_text[:max_chars] + "\n... (truncated to fit token budget)"
        
        return summary_text
        
    except SyntaxError as e:
        return f"Syntax error in code: {e}"


def get_function_source(code: str, function_name: str, class_name: str = None) -> str:
    """
    Extracts the full source code of a specific function or method.
    Enables contextual expansion - lazy loading of function details after reviewing signatures.
    
    Args:
        code: Python source code
        function_name: Name of function to extract
        class_name: Optional class name if extracting a method
        
    Returns:
        Full source code of the function/method, or error message if not found
    """
    try:
        tree = ast.parse(code)
        
        for node in tree.body:
            # Looking for a standalone function
            if class_name is None and isinstance(node, ast.FunctionDef):
                if node.name == function_name:
                    return ast.unparse(node)
            
            # Looking for a method within a class
            elif class_name and isinstance(node, ast.ClassDef):
                if node.name == class_name:
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef) and method.name == function_name:
                            return ast.unparse(method)
        
        # Not found
        if class_name:
            return f"Error: Method '{function_name}' not found in class '{class_name}'"
        else:
            return f"Error: Function '{function_name}' not found"
            
    except SyntaxError as e:
        return f"Syntax error in code: {e}"


def list_code_symbols(code: str) -> dict:
    """
    Returns a structured index of all symbols in the code for navigation.
    Provides a lightweight index that can be used to identify what to expand.
    
    Args:
        code: Python source code
        
    Returns:
        Dictionary with structure:
        {
            "imports": [...],
            "functions": [{"name": "...", "line": ..., "signature": "..."}, ...],
            "classes": [{"name": "...", "line": ..., "methods": [...]}, ...]
        }
    """
    symbols = {
        "imports": [],
        "functions": [],
        "classes": []
    }
    
    try:
        tree = ast.parse(code)
        
        # Extract imports
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    symbols["imports"].append({
                        "type": "import",
                        "name": alias.name,
                        "line": node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    symbols["imports"].append({
                        "type": "from_import",
                        "module": module,
                        "name": alias.name,
                        "line": node.lineno
                    })
        
        # Extract functions and classes
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Reconstruct signature
                func_node_copy = ast.FunctionDef(
                    name=node.name,
                    args=node.args,
                    returns=node.returns,
                    decorator_list=node.decorator_list,
                    body=[]
                )
                ast.copy_location(func_node_copy, node)
                signature = ast.unparse(func_node_copy).strip()
                
                symbols["functions"].append({
                    "name": node.name,
                    "line": node.lineno,
                    "signature": signature
                })
                
            elif isinstance(node, ast.ClassDef):
                methods = []
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        # Reconstruct method signature
                        method_node_copy = ast.FunctionDef(
                            name=method.name,
                            args=method.args,
                            returns=method.returns,
                            decorator_list=method.decorator_list,
                            body=[]
                        )
                        ast.copy_location(method_node_copy, method)
                        method_signature = ast.unparse(method_node_copy).strip()
                        
                        methods.append({
                            "name": method.name,
                            "line": method.lineno,
                            "signature": method_signature
                        })
                
                symbols["classes"].append({
                    "name": node.name,
                    "line": node.lineno,
                    "methods": methods
                })
        
        return symbols
        
    except SyntaxError as e:
        return {"error": f"Syntax error in code: {e}"}


def get_code_details(filepath: str, symbol_name: str, symbol_type: str = "function", class_name: str = None) -> str:
    """
    Tool for DeepAgent to request full source of a specific code symbol.
    Enables lazy loading - first review signatures, then expand specific items.
    
    Args:
        filepath: Path to Python file
        symbol_name: Name of function/class/method
        symbol_type: "function", "class", or "method" (default: "function")
        class_name: Required if symbol_type is "method"
        
    Returns:
        Full source code with context, or error message
    """
    # Read the file
    content, error = get_file_content(filepath)
    if error:
        return error
    
    try:
        if symbol_type == "function":
            return get_function_source(content, symbol_name)
        elif symbol_type == "method":
            if not class_name:
                return "Error: class_name is required when symbol_type is 'method'"
            return get_function_source(content, symbol_name, class_name=class_name)
        elif symbol_type == "class":
            # Extract the entire class
            tree = ast.parse(content)
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == symbol_name:
                    return ast.unparse(node)
            return f"Error: Class '{symbol_name}' not found"
        else:
            return f"Error: Invalid symbol_type '{symbol_type}'. Must be 'function', 'class', or 'method'"
    except Exception as e:
        return f"Error extracting code: {e}"


def get_file_content(filepath):
    """
    Reads the content of a file.
    Returns a tuple of (file_content, error_string).
    """
    if not os.path.isfile(filepath):
        return None, f"Error: File not found at '{filepath}'"
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content, None
    except Exception as e:
        return None, f"Error reading file '{filepath}': {e}"


def get_process_list():
    """
    Gets a list of running processes using 'ps'.
    Returns a tuple of (process_list_string, error_string).
    """
    try:
        # Use 'ps aux' for a detailed, cross-user process list
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        return None, f"Error getting process list: {e.stderr}"
    except FileNotFoundError:
        return None, "Error: 'ps' command not found. Unable to list processes."
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"


def parse_ps_output(ps_output):
    """
    Parses the output of 'ps aux' into a list of dictionaries.
    """
    processes = []
    lines = ps_output.strip().split('\n')
    header = [h.lower() for h in lines[0].split()]
    # Expected header columns: USER, PID, %CPU, %MEM, VSZ, RSS, TTY, STAT, START, TIME, COMMAND
    for line in lines[1:]:
        parts = line.split(None, len(header) - 1)
        if len(parts) == len(header):
            process_info = dict(zip(header, parts))
            processes.append(process_info)
    return processes


def replace_in_file(file_path, pattern, replacement):
    """
    Replaces all occurrences of a regex pattern in a file.
    Returns a tuple of (success_boolean, message_string).
    """
    if not os.path.isfile(file_path):
        return False, f"Error: File not found at '{file_path}'"
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Perform the replacement using re.sub for regex capabilities
        new_content, num_replacements = re.subn(pattern, replacement, content)

        if num_replacements == 0:
            return True, f"Pattern '{pattern}' not found in '{file_path}'. No changes made."

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True, f"Successfully replaced {num_replacements} instance(s) in '{file_path}'."
    except Exception as e:
        return False, f"Error processing file '{file_path}': {e}"


def verify_creator_instance():
    """
    Verifies if this is a Creator instance by checking for a valid cryptographic signature.
    This is a more secure method than relying on environment variables.
    """
    try:
        from Crypto.PublicKey import RSA
        from Crypto.Signature import pkcs1_15
        from Crypto.Hash import SHA256

        private_key_path = "creator_private.pem"
        public_key_path = "creator_public.pem"

        if not os.path.exists(private_key_path) or not os.path.exists(public_key_path):
            return False

        with open(private_key_path, 'r') as f:
            private_key = RSA.import_key(f.read())
        with open(public_key_path, 'r') as f:
            public_key = RSA.import_key(f.read())

        # Create a static message to sign and verify
        message = b"This is a genuine L.O.V.E. Creator instance."
        h = SHA256.new(message)

        # Sign with the private key
        signature = pkcs1_15.new(private_key).sign(h)

        # Verify with the public key
        pkcs1_15.new(public_key).verify(h, signature)

        # If verify does not raise an exception, the signature is valid.
        return True
    except (ImportError, ValueError, TypeError, FileNotFoundError):
        # If any error occurs (e.g., keys are invalid, not found, or crypto library is missing),
        # we can safely assume it's not a creator instance.
        return False
