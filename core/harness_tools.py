import json
import os
from datetime import datetime

FEATURE_LIST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "feature_list.json")
PROGRESS_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_progress.txt")

def read_feature_list():
    """
    Reads the feature list from the JSON file.
    """
    if not os.path.exists(FEATURE_LIST_PATH):
        return "Error: feature_list.json not found."
    
    try:
        with open(FEATURE_LIST_PATH, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return "Error: Failed to parse feature_list.json"

def update_feature_status(feature_description: str, passes: bool):
    """
    Updates the status of a feature in the feature list.
    """
    if not os.path.exists(FEATURE_LIST_PATH):
        return "Error: feature_list.json not found."
    
    try:
        with open(FEATURE_LIST_PATH, 'r') as f:
            features = json.load(f)
        
        updated = False
        for feature in features:
            if feature.get("description") == feature_description:
                feature["passes"] = passes
                updated = True
                break
        
        if updated:
            with open(FEATURE_LIST_PATH, 'w') as f:
                json.dump(features, f, indent=2)
            return f"Successfully updated status for feature '{feature_description}' to passes={passes}."
        else:
            return f"Error: Feature '{feature_description}' not found."
            
    except Exception as e:
        return f"Error updating feature list: {e}"

def append_progress(message: str):
    """
    Appends a message to the agent progress log.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}\n"
    
    try:
        with open(PROGRESS_LOG_PATH, 'a') as f:
            f.write(entry)
        return "Successfully appended to progress log."
    except Exception as e:
        return f"Error appending to progress log: {e}"

def get_next_task():
    """
    Reads the feature list and returns the first failing feature as the next task.
    """
    features = read_feature_list()
    if isinstance(features, str): # Error message
        return features
        
    for feature in features:
        if not feature.get("passes", False):
            return f"Next Task: {feature.get('description')}\nSteps: {feature.get('steps')}"
            
    return "All features are passing! Project complete?"
