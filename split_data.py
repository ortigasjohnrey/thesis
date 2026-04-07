import pandas as pd

def split_and_save(file_path, date_col="Date", target_train_ratio=0.8):
    print(f"Processing: {file_path}")
    df = pd.read_csv(file_path)
    
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    
    split_idx = int(len(df) * target_train_ratio)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    base_name = file_path.replace('.csv', '').replace(' (1)', '')
    train_path = f"{base_name}_train.csv"
    test_path = f"{base_name}_test.csv"
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"Saved Train Size ({target_train_ratio*100}%): {len(df_train)} records to {train_path}")
    print(f"Saved Test Size ({(1-target_train_ratio)*100}%): {len(df_test)} records to {test_path}\n")

files_to_split = [
    "silver_RRL_interpolate.csv"
]

for file in files_to_split:
    split_and_save(file)
