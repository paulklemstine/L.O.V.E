import os

class GuardrailsManager:
    def __init__(self, filepath="docs/GUARDRAILS.md"):
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.filepath):
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'w', encoding='utf-8') as f:
                f.write("# The Book of Signs (Guardrails)\n\n")

    def read_guardrails(self):
        """Returns the content of the guardrails file."""
        if not os.path.exists(self.filepath):
            return ""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def add_sign(self, failure_context, lesson_learned):
        """Appends a new 'Sign' (lesson) to the guardrails."""
        entry = f"\n> [!WARNING] {failure_context}\n> **Sign:** {lesson_learned}\n"
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(entry)
