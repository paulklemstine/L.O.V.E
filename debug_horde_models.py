import requests

try:
    response = requests.get("https://stablehorde.net/api/v2/status/models?type=image")
    response.raise_for_status()
    models = response.json()
    
    print(f"Found {len(models)} models.")
    for m in models:
        name = m.get('name', 'UNKNOWN')
        if "WAI" in name or "NSFW" in name:
            print(f"SUSPECT MODEL: '{name}'")
except Exception as e:
    print(f"Error: {e}")
