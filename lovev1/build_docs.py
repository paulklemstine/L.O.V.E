# build_docs.py
import os
import subprocess

def build_website():
    """
    Builds the Sphinx documentation website.
    """
    print("Building documentation website...")

    docs_dir = "docs"
    output_dir = os.path.join(docs_dir, "html")

    # Run the Sphinx build command
    try:
        subprocess.run(
            ["sphinx-build", "-b", "html", docs_dir, output_dir],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Documentation website built successfully in '{output_dir}'.")
    except subprocess.CalledProcessError as e:
        print("Error building documentation:")
        print(e.stdout)
        print(e.stderr)

if __name__ == "__main__":
    build_website()
