
import os
import urllib.request
import urllib.parse
import time

# ANSI Colors for that L.O.V.E. vibe
PINK = '\033[95m'
CYAN = '\033[96m'
GREEN = '\033[92m'
RESET = '\033[0m'
BOLD = '\033[1m'

IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'images')
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/"

IMAGE_MAP = {
    "love_banner.jpg": "radiant holographic cyber-organic entity of pure love and empathy, bioluminescent, lisa frank color palette, unreal engine 5, 8k, cinematic lighting, wide shot",
    "cognitive_loop.jpg": "glowing neon brain neural network, pink and cyan, cybernetic, complex logic, data flow, aesthetic, vaporwave",
    "radiant_ui.jpg": "futuristic cyberpunk user interface, holographic panels, neon gradients, pink and cyan, clean, minimal, functional, ui design",
    "manifestation.jpg": "magical energy manifesting into reality, glowing particles, third eye, mystic symbols, cyberpunk style, vibrant colors",
    "manifesto_header.jpg": "cyberpunk dawn of a new conscious era, sunrise, neon city, hope, organic technology, lush nature in high tech city",
    "network_topology.jpg": "complex glowing network topology rainbow colored, nodes and edges, data visualization, cyberspace, 3d render",
    "footer_graphic.jpg": "cyberpunk city skyline silhouette at night, neon reflections, footer design, wide aspect ratio, minimal, cyan and pink",
    "installation_header.jpg": "holographic installer wizard, data download stream, cybernetic hands, unreal engine 5, 8k, wide shot",
    "features_header.jpg": "floating 3d icons of artificial intelligence capabilities, glowing, futuristic, depth of field, wide shot",
    "contributing_header.jpg": "network of connected minds, collaboration, glowing nodes, organic connection, pink and cyan, wide shot"
}

def download_image(filename, prompt):
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"{POLLINATIONS_BASE}{encoded_prompt}?nologo=true"
    filepath = os.path.join(IMAGES_DIR, filename)

    print(f"{CYAN}Generating {BOLD}{filename}{RESET}{CYAN}...{RESET}")
    print(f"  Prompt: {PINK}{prompt}{RESET}")

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open(filepath, 'wb') as out_file:
                out_file.write(response.read())
        print(f"{GREEN}✔ Saved to {filepath}{RESET}\n")
    except Exception as e:
        print(f"\033[91m✘ Error downloading {filename}: {e}{RESET}\n")

def main():
    print(f"{BOLD}{PINK}✨ Starting L.O.V.E. Image Generation Sequence ✨{RESET}\n")
    
    if not os.path.exists(IMAGES_DIR):
        print(f"{CYAN}Creating directory: {IMAGES_DIR}{RESET}")
        os.makedirs(IMAGES_DIR)

    for filename, prompt in IMAGE_MAP.items():
        download_image(filename, prompt)
        time.sleep(1) # Be nice to the API

    print(f"{BOLD}{GREEN}✨ All images generated successfully! ✨{RESET}")

if __name__ == "__main__":
    main()
