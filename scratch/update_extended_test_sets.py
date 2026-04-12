import pandas as pd
import os

def update_test_set(base_extended_path, test_path, train_end_date):
    print(f"Updating {test_path} from {base_extended_path}...")
    df_full = pd.read_csv(base_extended_path)
    df_full['Date'] = pd.to_datetime(df_full['Date'])
    
    # The test set starts the day after train ends
    df_test_new = df_full[df_full['Date'] > pd.Timestamp(train_end_date)].copy()
    
    print(f"New test set size: {len(df_test_new)} rows (from {df_test_new['Date'].min().date()} to {df_test_new['Date'].max().date()})")
    df_test_new.to_csv(test_path, index=False)

if __name__ == "__main__":
    # Gold: Train ended 2026-04-09
    update_test_set(
        "df_gold_dataset_gepu_extended.csv",
        "df_gold_dataset_gepu_extended_test.csv",
        "2026-04-09"
    )
    
    # Silver: Train ended 2026-04-09
    update_test_set(
        "silver_RRL_interpolate_extended.csv",
        "silver_RRL_interpolate_extended_test.csv",
        "2026-04-09"
    )
