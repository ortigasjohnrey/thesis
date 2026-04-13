import json
import os

STATE_FILE = 'simulation_state.json'

def force_reset():
    if not os.path.exists(STATE_FILE):
        print("State file not found. Nothing to reset.")
        return
        
    with open(STATE_FILE, 'r') as f:
        data = json.load(f)
        
    for asset in data:
        data[asset]['diagnostic_logs'] = []
        data[asset]['history'] = {}
        data[asset]['test_idx'] = 0
        data[asset]['current_date'] = "2026-04-10"
        
    with open(STATE_FILE, 'w') as f:
        json.dump(data, f, indent=2)
        
    print("SUCCESS: Simulation memory has been completely wiped.")
    print("Please restart your API and refresh the dashboard.")

if __name__ == "__main__":
    force_reset()
