"""
Shell Executor for the Virtual Filesystem.

Executes shell commands (ls, cat, grep, mv, etc.) against the virtual filesystem.
This provides the primary interface for agents to interact with the system.
"""

import os
import re
import shlex
import fnmatch
import logging
from typing import Dict, List, Optional, Tuple, Any
from core.fuse.virtual_filesystem import VirtualFilesystem
from core.fuse.base import (
    FilesystemError,
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    PermissionError,
)

logger = logging.getLogger(__name__)


class ShellExecutor:
    """
    Executes shell commands against a VirtualFilesystem.
    
    Supports common Unix commands that agents are trained on:
    - ls: List directory contents
    - cat: Read file contents  
    - echo: Write content (with redirection)
    - find: Search for files
    - grep: Search within files
    - mv: Move/rename files
    - cp: Copy files
    - mkdir: Create directories
    - rm: Remove files
    - tree: Show directory tree
    - pwd: Print working directory
    - touch: Create empty file
    - head/tail: View first/last lines
    
    Example:
        vfs = VirtualFilesystem()
        shell = ShellExecutor(vfs)
        
        result = shell.execute("ls /tools")
        result = shell.execute("cat /tools/execute/schema.json")
        result = shell.execute("echo '{\"command\": \"pwd\"}' > /tools/execute/invoke")
    """
    
    def __init__(self, vfs: VirtualFilesystem):
        self.vfs = vfs
        self.cwd = "/scratch"  # Current working directory
        self.env: Dict[str, str] = {}  # Environment variables
        
        # Command handlers
        self._commands = {
            "ls": self._cmd_ls,
            "cat": self._cmd_cat,
            "echo": self._cmd_echo,
            "find": self._cmd_find,
            "grep": self._cmd_grep,
            "mv": self._cmd_mv,
            "cp": self._cmd_cp,
            "mkdir": self._cmd_mkdir,
            "rm": self._cmd_rm,
            "rmdir": self._cmd_rmdir,
            "tree": self._cmd_tree,
            "pwd": self._cmd_pwd,
            "cd": self._cmd_cd,
            "touch": self._cmd_touch,
            "head": self._cmd_head,
            "tail": self._cmd_tail,
            "wc": self._cmd_wc,
            "tee": self._cmd_tee,
        }
    
    def execute(self, command: str) -> Tuple[int, str, str]:
        """
        Execute a shell command.
        
        Args:
            command: Shell command string
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        logger.debug(f"Shell executing: {command}")
        
        try:
            # Handle pipes
            if "|" in command:
                return self._execute_pipeline(command)
            
            # Handle redirections
            stdout_file = None
            stdout_append = False
            stdin_content = None
            
            # Parse output redirection (>> or >)
            if ">>" in command:
                parts = command.split(">>", 1)
                command = parts[0].strip()
                stdout_file = parts[1].strip()
                stdout_append = True
            elif ">" in command and "2>" not in command:
                parts = command.split(">", 1)
                command = parts[0].strip()
                stdout_file = parts[1].strip()
            
            # Parse input redirection (<)
            if "<" in command:
                parts = command.split("<", 1)
                command = parts[0].strip()
                input_file = self._resolve_path(parts[1].strip())
                stdin_content = self.vfs.read(input_file)
            
            # Parse command and arguments
            try:
                tokens = shlex.split(command)
            except ValueError as e:
                return 1, "", f"Parse error: {e}"
            
            if not tokens:
                return 0, "", ""
            
            cmd_name = tokens[0]
            args = tokens[1:]
            
            # Execute command
            if cmd_name in self._commands:
                exit_code, stdout, stderr = self._commands[cmd_name](args, stdin_content)
            else:
                return 127, "", f"Command not found: {cmd_name}"
            
            # Handle output redirection
            if stdout_file and exit_code == 0:
                output_path = self._resolve_path(stdout_file)
                self.vfs.write(output_path, stdout, append=stdout_append)
                stdout = ""
            
            return exit_code, stdout, stderr
            
        except FilesystemError as e:
            return 1, "", str(e)
        except Exception as e:
            logger.exception(f"Shell execution error: {e}")
            return 1, "", f"Error: {e}"
    
    def _execute_pipeline(self, command: str) -> Tuple[int, str, str]:
        """Execute a pipeline of commands."""
        commands = [c.strip() for c in command.split("|")]
        stdout = ""
        
        for i, cmd in enumerate(commands):
            # For subsequent commands, pipe previous stdout as input
            if i > 0 and stdout:
                # Prepend echo with previous output
                # This is a simplification; real shell would use stdin
                cmd = f"echo {shlex.quote(stdout)} | {cmd}"
                exit_code, stdout, stderr = self.execute(cmd.split("|", 1)[1])
            else:
                exit_code, stdout, stderr = self.execute(cmd)
            
            if exit_code != 0:
                return exit_code, stdout, stderr
        
        return 0, stdout, stderr
    
    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to current working directory."""
        if path.startswith("/"):
            return os.path.normpath(path).replace("\\", "/")
        return os.path.normpath(os.path.join(self.cwd, path)).replace("\\", "/")
    
    def _cmd_ls(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """List directory contents."""
        show_long = "-l" in args or "-la" in args or "-al" in args
        show_all = "-a" in args or "-la" in args or "-al" in args
        
        # Remove flags from args
        paths = [a for a in args if not a.startswith("-")]
        if not paths:
            paths = [self.cwd]
        
        output_lines = []
        
        for path in paths:
            resolved = self._resolve_path(path)
            
            try:
                if self.vfs.isfile(resolved):
                    # ls on a file just shows the filename
                    if show_long:
                        attrs = self.vfs.getattr(resolved)
                        output_lines.append(self._format_ls_entry(os.path.basename(path), attrs))
                    else:
                        output_lines.append(os.path.basename(path))
                else:
                    entries = self.vfs.readdir(resolved)
                    
                    if not show_all:
                        entries = [e for e in entries if not e.startswith(".")]
                    
                    if len(paths) > 1:
                        output_lines.append(f"{path}:")
                    
                    for entry in sorted(entries):
                        entry_path = os.path.join(resolved, entry).replace("\\", "/")
                        if show_long:
                            try:
                                attrs = self.vfs.getattr(entry_path)
                                output_lines.append(self._format_ls_entry(entry, attrs))
                            except FileNotFoundError:
                                output_lines.append(f"?????????? ? {entry}")
                        else:
                            if self.vfs.isdir(entry_path):
                                output_lines.append(f"{entry}/")
                            else:
                                output_lines.append(entry)
                    
                    if len(paths) > 1:
                        output_lines.append("")
                        
            except FileNotFoundError:
                return 1, "", f"ls: cannot access '{path}': No such file or directory"
            except NotADirectoryError:
                return 1, "", f"ls: cannot access '{path}': Not a directory"
        
        return 0, "\n".join(output_lines), ""
    
    def _format_ls_entry(self, name: str, attrs) -> str:
        """Format a single ls -l entry."""
        import time
        
        mode = "d" if attrs.is_dir() else "-"
        mode += "rwxr-xr-x" if attrs.is_dir() else "rw-r--r--"
        
        size = attrs.size
        mtime = time.strftime("%b %d %H:%M", time.localtime(attrs.mtime))
        
        return f"{mode}  1 agent agent {size:8d} {mtime} {name}"
    
    def _cmd_cat(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Concatenate and display file contents."""
        if not args:
            if stdin:
                return 0, stdin, ""
            return 1, "", "cat: missing file operand"
        
        output_parts = []
        
        for path in args:
            resolved = self._resolve_path(path)
            try:
                content = self.vfs.read(resolved)
                output_parts.append(content)
            except FileNotFoundError:
                return 1, "", f"cat: {path}: No such file or directory"
            except IsADirectoryError:
                return 1, "", f"cat: {path}: Is a directory"
        
        return 0, "\n".join(output_parts), ""
    
    def _cmd_echo(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Echo text to stdout."""
        no_newline = "-n" in args
        args = [a for a in args if a != "-n"]
        
        output = " ".join(args)
        if not no_newline:
            output += ""  # Newline handled by join
        
        return 0, output, ""
    
    def _cmd_find(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Find files matching criteria."""
        # Parse arguments
        path = self.cwd
        name_pattern = None
        type_filter = None
        
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "-name" and i + 1 < len(args):
                name_pattern = args[i + 1]
                i += 2
            elif arg == "-type" and i + 1 < len(args):
                type_filter = args[i + 1]
                i += 2
            elif not arg.startswith("-"):
                path = arg
                i += 1
            else:
                i += 1
        
        resolved = self._resolve_path(path)
        results = []
        
        def search(current_path: str):
            try:
                entries = self.vfs.readdir(current_path)
                for entry in entries:
                    full_path = os.path.join(current_path, entry).replace("\\", "/")
                    is_dir = self.vfs.isdir(full_path)
                    
                    # Type filter
                    if type_filter:
                        if type_filter == "f" and is_dir:
                            continue
                        if type_filter == "d" and not is_dir:
                            continue
                    
                    # Name filter
                    if name_pattern:
                        if not fnmatch.fnmatch(entry, name_pattern):
                            if is_dir:
                                search(full_path)
                            continue
                    
                    results.append(full_path)
                    
                    if is_dir:
                        search(full_path)
                        
            except (FileNotFoundError, NotADirectoryError):
                pass
        
        search(resolved)
        return 0, "\n".join(results), ""
    
    def _cmd_grep(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Search for patterns in files."""
        case_insensitive = "-i" in args
        show_line_numbers = "-n" in args
        recursive = "-r" in args or "-R" in args
        
        # Remove flags
        args = [a for a in args if not a.startswith("-")]
        
        if not args:
            return 1, "", "grep: missing pattern"
        
        pattern = args[0]
        paths = args[1:] if len(args) > 1 else [self.cwd]
        
        if case_insensitive:
            regex = re.compile(pattern, re.IGNORECASE)
        else:
            regex = re.compile(pattern)
        
        results = []
        
        def grep_file(file_path: str, show_filename: bool = False):
            try:
                content = self.vfs.read(file_path)
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        prefix = ""
                        if show_filename:
                            prefix = f"{file_path}:"
                        if show_line_numbers:
                            prefix += f"{i}:"
                        results.append(f"{prefix}{line}")
            except (FileNotFoundError, IsADirectoryError):
                pass
        
        def grep_recursive(dir_path: str):
            try:
                for entry in self.vfs.readdir(dir_path):
                    full_path = os.path.join(dir_path, entry).replace("\\", "/")
                    if self.vfs.isdir(full_path):
                        grep_recursive(full_path)
                    else:
                        grep_file(full_path, show_filename=True)
            except (FileNotFoundError, NotADirectoryError):
                pass
        
        show_filename = len(paths) > 1 or recursive
        
        for path in paths:
            resolved = self._resolve_path(path)
            
            if self.vfs.isdir(resolved):
                if recursive:
                    grep_recursive(resolved)
                else:
                    return 1, "", f"grep: {path}: Is a directory"
            else:
                grep_file(resolved, show_filename)
        
        if results:
            return 0, "\n".join(results), ""
        return 1, "", ""  # No matches
    
    def _cmd_mv(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Move/rename files."""
        if len(args) < 2:
            return 1, "", "mv: missing destination file operand"
        
        sources = args[:-1]
        dest = self._resolve_path(args[-1])
        
        for src in sources:
            src_resolved = self._resolve_path(src)
            
            try:
                # If dest is a directory, move into it
                if self.vfs.isdir(dest):
                    new_name = os.path.basename(src_resolved)
                    dest_path = os.path.join(dest, new_name).replace("\\", "/")
                else:
                    dest_path = dest
                
                self.vfs.rename(src_resolved, dest_path)
            except FilesystemError as e:
                return 1, "", f"mv: {e}"
        
        return 0, "", ""
    
    def _cmd_cp(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Copy files."""
        recursive = "-r" in args or "-R" in args
        args = [a for a in args if not a.startswith("-")]
        
        if len(args) < 2:
            return 1, "", "cp: missing destination file operand"
        
        sources = args[:-1]
        dest = self._resolve_path(args[-1])
        
        for src in sources:
            src_resolved = self._resolve_path(src)
            
            try:
                if self.vfs.isdir(src_resolved):
                    if not recursive:
                        return 1, "", f"cp: -r not specified; omitting directory '{src}'"
                    # Recursive copy not fully implemented
                    return 1, "", "cp: recursive copy not yet implemented"
                
                content = self.vfs.read(src_resolved)
                
                if self.vfs.isdir(dest):
                    dest_path = os.path.join(dest, os.path.basename(src_resolved)).replace("\\", "/")
                else:
                    dest_path = dest
                
                self.vfs.write(dest_path, content)
                
            except FilesystemError as e:
                return 1, "", f"cp: {e}"
        
        return 0, "", ""
    
    def _cmd_mkdir(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Create directories."""
        parents = "-p" in args
        args = [a for a in args if not a.startswith("-")]
        
        if not args:
            return 1, "", "mkdir: missing operand"
        
        for path in args:
            resolved = self._resolve_path(path)
            try:
                self.vfs.mkdir(resolved)
            except FilesystemError as e:
                if not parents:
                    return 1, "", f"mkdir: {e}"
        
        return 0, "", ""
    
    def _cmd_rm(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Remove files."""
        recursive = "-r" in args or "-R" in args or "-rf" in args
        force = "-f" in args or "-rf" in args
        args = [a for a in args if not a.startswith("-")]
        
        if not args:
            return 1, "", "rm: missing operand"
        
        for path in args:
            resolved = self._resolve_path(path)
            try:
                if self.vfs.isdir(resolved):
                    if not recursive:
                        return 1, "", f"rm: cannot remove '{path}': Is a directory"
                    self.vfs.rmdir(resolved)
                else:
                    self.vfs.remove(resolved)
            except FileNotFoundError:
                if not force:
                    return 1, "", f"rm: cannot remove '{path}': No such file or directory"
            except FilesystemError as e:
                return 1, "", f"rm: {e}"
        
        return 0, "", ""
    
    def _cmd_rmdir(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Remove empty directories."""
        args = [a for a in args if not a.startswith("-")]
        
        if not args:
            return 1, "", "rmdir: missing operand"
        
        for path in args:
            resolved = self._resolve_path(path)
            try:
                self.vfs.rmdir(resolved)
            except FilesystemError as e:
                return 1, "", f"rmdir: {e}"
        
        return 0, "", ""
    
    def _cmd_tree(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Display directory tree."""
        path = args[0] if args else self.cwd
        resolved = self._resolve_path(path)
        
        try:
            tree_output = self.vfs.tree(resolved)
            header = resolved
            return 0, f"{header}\n{tree_output}", ""
        except FilesystemError as e:
            return 1, "", f"tree: {e}"
    
    def _cmd_pwd(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Print working directory."""
        return 0, self.cwd, ""
    
    def _cmd_cd(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Change directory."""
        if not args:
            self.cwd = "/scratch"
            return 0, "", ""
        
        path = self._resolve_path(args[0])
        
        if not self.vfs.exists(path):
            return 1, "", f"cd: {args[0]}: No such file or directory"
        if not self.vfs.isdir(path):
            return 1, "", f"cd: {args[0]}: Not a directory"
        
        self.cwd = path
        return 0, "", ""
    
    def _cmd_touch(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Create empty files or update timestamps."""
        if not args:
            return 1, "", "touch: missing file operand"
        
        for path in args:
            resolved = self._resolve_path(path)
            try:
                if not self.vfs.exists(resolved):
                    self.vfs.write(resolved, "")
            except FilesystemError as e:
                return 1, "", f"touch: {e}"
        
        return 0, "", ""
    
    def _cmd_head(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Display first lines of a file."""
        n_lines = 10
        
        # Parse -n argument
        i = 0
        while i < len(args):
            if args[i] == "-n" and i + 1 < len(args):
                try:
                    n_lines = int(args[i + 1])
                except ValueError:
                    return 1, "", f"head: invalid number of lines: '{args[i + 1]}'"
                args = args[:i] + args[i + 2:]
            else:
                i += 1
        
        if not args:
            if stdin:
                lines = stdin.splitlines()[:n_lines]
                return 0, "\n".join(lines), ""
            return 1, "", "head: missing file operand"
        
        path = self._resolve_path(args[0])
        try:
            content = self.vfs.read(path)
            lines = content.splitlines()[:n_lines]
            return 0, "\n".join(lines), ""
        except FilesystemError as e:
            return 1, "", f"head: {e}"
    
    def _cmd_tail(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Display last lines of a file."""
        n_lines = 10
        
        i = 0
        while i < len(args):
            if args[i] == "-n" and i + 1 < len(args):
                try:
                    n_lines = int(args[i + 1])
                except ValueError:
                    return 1, "", f"tail: invalid number of lines: '{args[i + 1]}'"
                args = args[:i] + args[i + 2:]
            else:
                i += 1
        
        if not args:
            if stdin:
                lines = stdin.splitlines()[-n_lines:]
                return 0, "\n".join(lines), ""
            return 1, "", "tail: missing file operand"
        
        path = self._resolve_path(args[0])
        try:
            content = self.vfs.read(path)
            lines = content.splitlines()[-n_lines:]
            return 0, "\n".join(lines), ""
        except FilesystemError as e:
            return 1, "", f"tail: {e}"
    
    def _cmd_wc(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Word, line, and byte count."""
        show_lines = "-l" in args
        show_words = "-w" in args
        show_chars = "-c" in args
        
        if not (show_lines or show_words or show_chars):
            show_lines = show_words = show_chars = True
        
        args = [a for a in args if not a.startswith("-")]
        
        if not args:
            if stdin:
                content = stdin
                filename = ""
            else:
                return 1, "", "wc: missing file operand"
        else:
            path = self._resolve_path(args[0])
            try:
                content = self.vfs.read(path)
                filename = args[0]
            except FilesystemError as e:
                return 1, "", f"wc: {e}"
        
        lines = len(content.splitlines())
        words = len(content.split())
        chars = len(content)
        
        parts = []
        if show_lines:
            parts.append(f"{lines:8d}")
        if show_words:
            parts.append(f"{words:8d}")
        if show_chars:
            parts.append(f"{chars:8d}")
        
        result = " ".join(parts)
        if filename:
            result += f" {filename}"
        
        return 0, result, ""
    
    def _cmd_tee(self, args: List[str], stdin: Optional[str] = None) -> Tuple[int, str, str]:
        """Read from stdin and write to both stdout and files."""
        append = "-a" in args
        args = [a for a in args if not a.startswith("-")]
        
        if stdin is None:
            return 1, "", "tee: requires stdin input"
        
        for path in args:
            resolved = self._resolve_path(path)
            try:
                self.vfs.write(resolved, stdin, append=append)
            except FilesystemError as e:
                return 1, stdin, f"tee: {e}"
        
        return 0, stdin, ""
