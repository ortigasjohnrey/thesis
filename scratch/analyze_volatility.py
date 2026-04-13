import json
import numpy as np

def analyze_magnitude():
    with open('simulation_state.json', 'r') as f:
        data = json.load(f)
    
    logs = data['gold']['diagnostic_logs']
    if len(logs) < 10:
        print("Waiting for more history...")
        return
    
    # Calculate Mean Absolute Return (Volatility)
    pr = np.abs([l['pred_ret'] for l in logs])
    ar = np.abs([l['actual_ret'] for l in logs])
    
    avg_pred = np.mean(pr)
    avg_actual = np.mean(ar)
    gap = avg_actual / (avg_pred + 1e-9)
    
    print(f"Average Model Volatility: {avg_pred:.5f}")
    print(f"Average Market Volatility: {avg_actual:.5f}")
    print(f"VOLATILITY GAP: {gap:.2f}x")
    
    if gap > 2.0:
        print("VERDICT: COWARDLY MODEL. The model is under-predicting the scale of moves.")
    elif gap < 0.5:
        print("VERDICT: HYPERACTIVE MODEL. The model is over-reacting to noise.")
    else:
        print("VERDICT: SCALE ALIGNED.")

if __name__ == "__main__":
    analyze_magnitude()
