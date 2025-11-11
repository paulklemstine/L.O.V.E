def overwrite_file(file_path, new_content):
    """
    Writes new_content to the file at file_path, overwriting existing content.

    Args:
        file_path (str): The path to the file.
        new_content (str): The new content to write to the file.
    """
    try:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"Successfully overwrote {file_path}")
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")
