## 2026-02-05 - [Regex Escaping in Raw Strings]
**Learning:** `re.findall(r'\\b', text)` matches a literal backslash followed by 'b', NOT a word boundary. The double backslash escapes the backslash even in raw strings. This bug caused the tokenizer to silently return empty lists, masking the true performance cost of the module.
**Action:** Use `r'\b'` for word boundaries in regex. Always verify performance optimization baselines with correctness tests first.
