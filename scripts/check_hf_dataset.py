from huggingface_hub import list_datasets

try:
    print("Searching for datasets...")
    datasets = list(list_datasets(search="opencompass", limit=20))
    for d in datasets:
        print(d.id)
    
    print("\nSpecific check for Open LMM Reasoning Leaderboard...")
    # Trying to guess the ID
    candidates = [
        "opencompass/Open_LMM_Reasoning_Leaderboard",
        "opencompass/open_lmm_reasoning_leaderboard",
        "opencompass/leaderboard",
        "opencompass/open_vlm_leaderboard"
    ]
    for c in candidates:
        print(f"Checking {c}...")
        try:
             # Just listing doesn't verify content, but existence
             # We can try to get info
             from huggingface_hub import dataset_info
             info = dataset_info(c)
             print(f"FOUND: {c}")
             print(info)
        except Exception as e:
             print(f"Not found or error: {c} ({e})")

except Exception as e:
    print(f"Error: {e}")
