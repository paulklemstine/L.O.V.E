# build_docs.py
import os
import markdown
import re

def build_website():
    """
    Builds a simple HTML website from the Markdown files in the docs/ directory,
    correctly handling links between the markdown files.
    """
    print("Building documentation website...")

    docs_dir = "docs"
    output_dir = os.path.join(docs_dir, "html")
    os.makedirs(output_dir, exist_ok=True)

    # Find all Markdown files in the docs/ directory (excluding subdirectories)
    md_files = [f for f in os.listdir(docs_dir) if f.endswith(".md") and os.path.isfile(os.path.join(docs_dir, f))]

    for md_file in md_files:
        input_path = os.path.join(docs_dir, md_file)
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
            # Convert markdown to HTML
            html_content = markdown.markdown(text, extensions=['fenced_code', 'tables'])

            # Find all markdown links and replace .md with .html
            # This regex looks for href attributes pointing to .md files
            html_content = re.sub(r'href="([^"]*)\.md"', r'href="\1.html"', html_content)

            # Add a basic HTML shell with some styling
            final_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{os.path.splitext(md_file)[0].replace('_', ' ').title()}</title>
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: 0 auto; background-color: #0d1117; color: #c9d1d9; }}
                    h1, h2, h3 {{ border-bottom: 1px solid #30363d; padding-bottom: 0.3em; }}
                    a {{ color: #58a6ff; text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                    code {{ background-color: #161b22; padding: 0.2em 0.4em; margin: 0; font-size: 85%; border-radius: 6px; }}
                    pre > code {{ display: block; padding: 10px; overflow-x: auto; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """

        # Create the HTML file
        html_file_name = os.path.splitext(md_file)[0] + ".html"
        output_path = os.path.join(output_dir, html_file_name)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_html)

    print(f"Documentation website built successfully in '{output_dir}'.")

if __name__ == "__main__":
    build_website()
