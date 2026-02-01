import os

def verify_a11y():
    filepath = 'core/web/static/index.html'
    if not os.path.exists(filepath):
        print(f"FAILED: {filepath} not found.")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    errors = []

    # Check #logs-header attributes
    if 'id="logs-header"' in content:
        if 'role="button"' not in content.split('id="logs-header"')[1].split('>')[0] and \
           'role="button"' not in content.split('id="logs-header"')[0].split('<div')[-1]:
             # This simple split might be flaky if attributes are in different order or multiline
             # Let's do a more robust check around the specific tag
             pass

    # A simpler check: Does the file contain the specific strings we expect in the context of the logs header?
    # We look for the logs-header div definition
    import re
    logs_header_match = re.search(r'<div[^>]*id="logs-header"[^>]*>', content)
    if logs_header_match:
        header_tag = logs_header_match.group(0)
        if 'role="button"' not in header_tag:
            errors.append("logs-header missing role='button'")
        if 'tabindex="0"' not in header_tag:
            errors.append("logs-header missing tabindex='0'")
        if 'aria-expanded="' not in header_tag:
            errors.append("logs-header missing aria-expanded attribute")
        if 'onkeydown="handleHeaderKey(event)"' not in header_tag:
            errors.append("logs-header missing onkeydown handler")
    else:
        errors.append("Could not find #logs-header div")

    # Check #chat-input label
    chat_input_match = re.search(r'<input[^>]*id="chat-input"[^>]*>', content)
    if chat_input_match:
        input_tag = chat_input_match.group(0)
        if 'aria-label=' not in input_tag:
            errors.append("chat-input missing aria-label")
    else:
        errors.append("Could not find #chat-input")

    # Check JS functions
    if 'function handleHeaderKey(e)' not in content and 'function handleHeaderKey(event)' not in content:
        errors.append("handleHeaderKey function not defined")

    if 'setAttribute(\'aria-expanded\'' not in content and 'setAttribute("aria-expanded"' not in content:
         # Check inside toggleLogs
         toggle_logs_loc = content.find('function toggleLogs()')
         if toggle_logs_loc != -1:
             toggle_body = content[toggle_logs_loc:toggle_logs_loc+500] # Grab enough context
             if 'aria-expanded' not in toggle_body:
                 errors.append("toggleLogs does not toggle aria-expanded")
         else:
             errors.append("toggleLogs function not found")

    if errors:
        for e in errors:
            print(f"FAILED: {e}")
        return False

    print("SUCCESS: All accessibility checks passed.")
    return True

if __name__ == "__main__":
    success = verify_a11y()
    if not success:
        exit(1)
