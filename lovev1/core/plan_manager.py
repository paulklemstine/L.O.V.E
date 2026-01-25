import re
import os

class PlanManager:
    def __init__(self, plan_path="IMPLEMENTATION_PLAN.md"):
        self.plan_path = plan_path

    def _read_lines(self):
        if not os.path.exists(self.plan_path):
            raise FileNotFoundError(f"Plan file not found: {self.plan_path}")
        with open(self.plan_path, 'r', encoding='utf-8') as f:
            return f.readlines()

    def _write_lines(self, lines):
        with open(self.plan_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    def get_next_task(self):
        """
        Finds the first unchecked task under '## Current Objectives'.
        Returns a dict with 'line_index' and 'text', or None.
        """
        lines = self._read_lines()
        in_objectives = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "## Current Objectives":
                in_objectives = True
                continue
            if stripped.startswith("## ") and in_objectives:
                break # Left the section
            
            if in_objectives:
                # Look for unchecked items: "- [ ] "
                match = re.match(r"^\s*-\s*\[\s*\]\s+(.*)", line)
                if match:
                    return {
                        "line_index": i,
                        "text": match.group(1).strip(), 
                        "raw_line": line
                    }
        return None

    def mark_task_complete(self, line_index):
        """Marks the task at line_index as [x]."""
        lines = self._read_lines()
        if line_index < 0 or line_index >= len(lines):
            return False
            
        line = lines[line_index]
        # Replace [ ] or [/] with [x]
        new_line = re.sub(r"\[\s*\]", "[x]", line, 1)
        if new_line == line:
             new_line = re.sub(r"\[/\]", "[x]", line, 1)
             
        lines[line_index] = new_line
        self._write_lines(lines)
        return True

    def mark_task_in_progress(self, line_index):
        """Marks the task at line_index as [/]."""
        lines = self._read_lines()
        if line_index < 0 or line_index >= len(lines):
            return False
            
        line = lines[line_index]
        new_line = re.sub(r"\[\s*\]", "[/]", line, 1)
        lines[line_index] = new_line
        self._write_lines(lines)
        return True

    def add_task(self, text, position="top"):
        """Adds a new task to ## Current Objectives. position='top' or 'bottom'."""
        lines = self._read_lines()
        insert_idx = -1
        
        # Find start of Current Objectives
        for i, line in enumerate(lines):
            if line.strip() == "## Current Objectives":
                insert_idx = i + 1
                break
        
        if insert_idx == -1:
            # Create section if missing
            lines.append("\n## Current Objectives\n")
            insert_idx = len(lines)
            
        new_line = f"- [ ] {text}\n"
        
        if position == "top":
            lines.insert(insert_idx, new_line)
        else:
            # Find end of section (next ## or EOF)
            j = insert_idx
            while j < len(lines):
                if lines[j].strip().startswith("## "):
                    break
                j += 1
            lines.insert(j, new_line)
            
        self._write_lines(lines)
        return True
