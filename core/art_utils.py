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
             
        # Use a monospace font to properly render ANSI art
        console.save_svg(
            svg_path, 
            title=filename_prefix,
            font_aspect_ratio=0.6
        )
        
        # Convert to PNG using PIL with a proper monospace font
        png_path = os.path.join(output_dir, f"{base_filename}.png")
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Get plain text from content
            if isinstance(art_content, Text):
                text_obj = art_content
            else:
                text_obj = Text.from_ansi(raw_content)
            
            plain_text = text_obj.plain
            lines = plain_text.split('\n')
            
            # Try to find a monospace font
            monospace_fonts = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
                "C:/Windows/Fonts/consola.ttf",
                "C:/Windows/Fonts/cour.ttf",
                "/System/Library/Fonts/Monaco.ttf",
            ]
            
            font = None
            font_size = 16
            for font_path in monospace_fonts:
                try:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, font_size)
                        break
                except Exception:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
                font_size = 10
            
            # Calculate image dimensions based on monospace character size
            char_width = font_size * 0.6
            char_height = font_size * 1.4
            max_line_len = max(len(line) for line in lines) if lines else 1
            
            img_width = int(max_line_len * char_width) + 20
            img_height = int(len(lines) * char_height) + 20
            
            # Create image with dark background
            img = Image.new('RGB', (img_width, img_height), color=(41, 41, 41))
            draw = ImageDraw.Draw(img)
            
            # Draw each line of text with default color
            y = 10
            for line in lines:
                draw.text((10, y), line, font=font, fill=(197, 200, 198))
                y += char_height
            
            img.save(png_path, 'PNG')
            core.logging.log_event(f"Saved artwork to {ansi_path}, {svg_path}, and {png_path}", "INFO")
            return ansi_path, svg_path, png_path
            
        except ImportError as e:
            core.logging.log_event(f"PIL not found, trying cairosvg. {e}", "WARNING")
            try:
                import cairosvg
                cairosvg.svg2png(url=svg_path, write_to=png_path)
                core.logging.log_event(f"Saved artwork to {ansi_path}, {svg_path}, and {png_path}", "INFO")
                return ansi_path, svg_path, png_path
            except ImportError:
                core.logging.log_event("cairosvg not found. Skipping PNG conversion.", "WARNING")
                return ansi_path, svg_path, None
        except Exception as e:
            core.logging.log_event(f"Failed to convert to PNG: {e}", "ERROR")
            return ansi_path, svg_path, None

    except Exception as e:
        core.logging.log_event(f"Failed to save artwork: {e}", "ERROR")
        return None, None, None

