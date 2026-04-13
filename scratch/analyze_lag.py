import json
import numpy as np
import pandas as pd

def analyze_lag():
    with open('simulation_state.json', 'r') as f:
        data = json.load(f)
    
    logs = data['gold']['diagnostic_logs']
    if len(logs) < 15:
        print("Waiting for more history (need 15+ days)...")
        return
    
    pr = np.array([l['pred_ret'] for l in logs])
    ar = np.array([l['actual_ret'] for l in logs])
    
    # Correlation between pred[t] and actual[t] (Direct Lead)
    c_direct = np.corrcoef(pr, ar)[0,1]
    
    # Correlation between pred[t] and actual[t-1] (Lag / Shadowing)
    c_lag = np.corrcoef(pr[1:], ar[:-1])[0,1]
    
    print(f"Direct Prediction Accuracy (Corr): {c_direct:.3f}")
    print(f"Lag Shadowing Factor (Corr[t, t-1]): {c_lag:.3f}")
    
    if c_lag > c_direct:
        print("VERDICT: RETRACING DETECTED. The model is following the price, not leading it.")
        print(f"Shadow Power: {(c_lag/c_direct if c_direct > 0 else 0):.2f}x stronger than lead.")
    else:
        print("VERDICT: PROACTIVE. The model is leading the market moves.")

if __name__ == "__main__":
    analyze_lag()
