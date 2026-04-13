import json
import numpy as np
import pandas as pd

def analyze_laziness():
    with open('simulation_state.json', 'r') as f:
        data = json.load(f)
    
    h = data['gold']['history']
    sorted_keys = sorted(h.keys())
    
    # Load actual prices for the same window to compare returns
    test_df = pd.read_csv('df_gold_dataset_gepu_extended_lively.csv')
    test_df['Date'] = pd.to_datetime(test_df['Date']).dt.date
    
    act_rets = []
    pred_rets = []
    
    for i, k in enumerate(sorted_keys):
        dt = pd.to_datetime(k).date()
        match = test_df[test_df['Date'] == dt]
        if not match.empty:
            idx = int(match.index[0])
            if idx > 0:
                prev_p = float(test_df.iloc[idx-1]['Gold_Futures'])
                act_p = float(match.iloc[0]['Gold_Futures'])
                pred_p = float(h[k])
                
                # Magnitude of returns
                act_rets.append(abs((act_p - prev_p) / prev_p))
                pred_rets.append(abs((pred_p - prev_p) / prev_p))
    
    if not act_rets:
        print("No paired data found.")
        return
        
    avg_act = np.mean(act_rets)
    avg_pred = np.mean(pred_rets)
    ratio = avg_pred / avg_act if avg_act > 0 else 0
    
    print(f"Avg Actual Absolute Return: {avg_act*100:.3f}%")
    print(f"Avg Prediction Absolute Return: {avg_pred*100:.3f}%")
    print(f"Magnitude Ratio (Pred/Act): {ratio:.3f}")
    
    if ratio < 0.4:
        print("VERDICT: HIGH LAZINESS DETECTED. Model is only predicting ~1/3 of market intensity.")
    elif ratio < 0.7:
        print("VERDICT: MODERATE LAZINESS. Model is too conservative.")
    else:
        print("VERDICT: STABLE. Magnitude matches current regime.")

if __name__ == "__main__":
    analyze_laziness()
