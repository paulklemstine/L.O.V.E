
import os

with open("love.log", "rb") as f:
    f.seek(0, 2)
    filesize = f.tell()
    # Read last 5MB or whole file
    read_size = min(filesize, 5 * 1024 * 1024)
    f.seek(filesize - read_size)
    
    # helper to handle decoding
    content = f.read().decode('utf-8', errors='ignore')
    lines = content.splitlines()
    
    keywords = [
        "Director generating", 
        "Image generation successful", 
        "Publishing post", 
        "Applying manual text overlay", 
        "Generating image with prompt", 
        "Provider", 
        "subliminal",
        "Aborting post"
    ]
    
    matches = []
    for line in lines:
        if any(x in line for x in keywords):
            matches.append(line)
            
    # Print last 30 matches
    for m in matches[-30:]:
        print(m)
