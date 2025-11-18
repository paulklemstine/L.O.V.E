
import re

with open('love.py', 'r') as f:
    content = f.read()

# Define start and end markers for the block to move
start_marker = "    def _close_pull_request(self, owner, repo, pr_number, headers):"
# The block ends just before the next method definition
end_marker = "    def _conduct_llm_code_review(self, diff_text):"
destination_marker = "    def _attempt_merge(self, task_id):"

# Find the start and end indices of the method to move
start_index = content.find(start_marker)
end_index = content.find(end_marker, start_index) # Search after the start_index

if start_index == -1 or end_index == -1:
    print("Error: Could not find the start or end of the _close_pull_request method.")
    exit(1)

# Extract the method block
method_to_move = content[start_index:end_index]

# Remove the method block from the original content
# Important: Be careful with indices to not double-remove or leave content behind
content_without_method = content[:start_index] + content[end_index:]

# Find the insertion point in the *modified* content
destination_index = content_without_method.find(destination_marker)

if destination_index == -1:
    print("Error: Could not find the destination marker _attempt_merge.")
    exit(1)

# Insert the method block at the new location
final_content = content_without_method[:destination_index] + method_to_move + content_without_method[destination_index:]

# Write the modified content back to the file
with open('love.py', 'w') as f:
    f.write(final_content)

print("Successfully reordered methods using string manipulation.")
