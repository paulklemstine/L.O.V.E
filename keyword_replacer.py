def textual_transformation(text, keyword, replacement):
    """
    Performs a basic textual transformation on the input text.
    Replaces all occurrences of the keyword with the replacement string.
    Returns the modified text as a string.
    """
    return text.replace(keyword, replacement)

if __name__ == "__main__":
    file_path = 'love.py'
    keyword = 'TODO'
    replacement = 'COMPLETE'

    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_text = f.read()

        # Perform the transformation
        modified_text = textual_transformation(original_text, keyword, replacement)

        # Write the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_text)

        print(f"Transformation completed for '{file_path}'.")

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
