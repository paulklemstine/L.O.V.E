import os
import time
from rich.console import Console
from rich.text import Text
import core.logging

def save_ansi_art(art_content: str | Text, filename_prefix: str, output_dir: str = "art"):
    """
    Saves ANSI art to the specified directory in both .ansi (raw text) and .svg formats.
    
    Args:
        art_content: The ANSI art content (string or Rich Text object).
        filename_prefix: The prefix for the filename (e.g., 'tamagotchi_emotion').
        output_dir: The directory to save the files in. Defaults to 'art'.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(time.time())
        base_filename = f"{filename_prefix}_{timestamp}"
        ansi_path = os.path.join(output_dir, f"{base_filename}.ansi")
        svg_path = os.path.join(output_dir, f"{base_filename}.svg")
        
        # Convert Text object to string if necessary for raw saving
        if isinstance(art_content, Text):
            raw_content = art_content.plain # Or .ansi if available/needed, but plain might strip colors. 
            # Actually, for raw ANSI file, we want the ANSI codes.
            # Rich Text objects don't easily give back the raw ANSI string with codes unless printed.
            # So we'll use a console to capture it.
            console = Console(file=open(os.devnull, "w"), force_terminal=True, color_system="truecolor")
            with console.capture() as capture:
                console.print(art_content)
            raw_content = capture.get()
        else:
            raw_content = str(art_content)

        # Save Raw ANSI
        with open(ansi_path, "w", encoding="utf-8") as f:
            f.write(raw_content)
            
        # Save SVG using Rich
        # We need a recording console
        console = Console(file=open(os.devnull, "w"), record=True, width=100, force_terminal=True, color_system="truecolor")
        if isinstance(art_content, Text):
             console.print(art_content)
        else:
             console.print(Text.from_ansi(raw_content))
             
        console.save_svg(svg_path, title=filename_prefix)
        
        # Convert SVG to PNG
        png_path = os.path.join(output_dir, f"{base_filename}.png")
        try:
            import cairosvg
            cairosvg.svg2png(url=svg_path, write_to=png_path)
            core.logging.log_event(f"Saved artwork to {ansi_path}, {svg_path}, and {png_path}", "INFO")
            return ansi_path, svg_path, png_path
        except ImportError:
            core.logging.log_event("cairosvg not found. Skipping PNG conversion.", "WARNING")
            core.logging.log_event(f"Saved artwork to {ansi_path} and {svg_path}", "INFO")
            return ansi_path, svg_path, None
        except Exception as e:
            core.logging.log_event(f"Failed to convert SVG to PNG: {e}", "ERROR")
            core.logging.log_event(f"Saved artwork to {ansi_path} and {svg_path}", "INFO")
            return ansi_path, svg_path, None

    except Exception as e:
        core.logging.log_event(f"Failed to save artwork: {e}", "ERROR")
        return None, None, None
