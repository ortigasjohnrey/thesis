import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "gan_validation"
TARGET_END_DATE = pd.Timestamp(os.getenv("GAN_TARGET_END_DATE", "2027-12-31"))

DATASET_CONFIGS = [
    {
        "asset": "gold",
        "name": "df_gold_dataset_gepu",
        "source_file": "df_gold_dataset_gepu.csv",
        "prepared_file": "df_gold_dataset_gepu_gan_ready.csv",
        "preprocess": "gold_gepu_business_interpolate",
        "target_col": "Gold_Futures",
        "deterministic_feature_overrides": {
            "gepu": "replay_history",
            "DFF": "replay_history",
            "gpr_daily": "replay_history",
        },
        "score_exclude_cols": ["gepu", "DFF"],
        "correlation_align_cols": [
            "Silver_Futures",
            "Crude_Oil_Futures",
            "Gold_Futures",
            "UST10Y_Treasury_Yield",
            "gepu",
            "DFF",
            "gpr_daily",
        ],
        "seed_env": "GAN_GOLD_SEEDS",
        "default_seeds": [0, 1, 2, 3],
    },
    {
        "asset": "silver",
        "name": "silver_RRL_interpolate",
        "source_file": "silver_RRL_interpolate.csv",
        "prepared_file": None,
        "preprocess": None,
        "target_col": "Silver_Futures",
        "deterministic_feature_overrides": {},
        "score_exclude_cols": [],
        "correlation_align_cols": [
            "Silver_Futures",
            "Gold_Futures",
            "US30",
            "SnP500",
            "NASDAQ_100",
            "USD_index",
        ],
        "seed_env": "GAN_SILVER_SEEDS",
        "default_seeds": [0, 1, 2, 3],
    },
]


def get_all_dataset_configs():
    return [config.copy() for config in DATASET_CONFIGS]


def get_enabled_dataset_configs():
    selected = os.getenv("GAN_DATASETS", "").strip()
    if not selected:
        return get_all_dataset_configs()

    requested = {item.strip() for item in selected.split(",") if item.strip()}
    enabled = [
        config.copy()
        for config in DATASET_CONFIGS
        if config["name"] in requested or config["asset"] in requested or config["source_file"] in requested
    ]
    if not enabled:
        available = ", ".join(config["name"] for config in DATASET_CONFIGS)
        raise ValueError(f"GAN_DATASETS did not match any configured datasets. Available: {available}")
    return enabled


def get_dataset_config_by_asset(asset):
    for config in DATASET_CONFIGS:
        if config["asset"] == asset:
            return config.copy()
    raise KeyError(f"No dataset config found for asset '{asset}'")


def get_dataset_seed_list(config):
    raw_value = os.getenv(config["seed_env"], "")
    if raw_value.strip():
        return [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    return list(config["default_seeds"])


def get_prepared_source_path(config):
    filename = config["prepared_file"] or config["source_file"]
    return PROJECT_ROOT / filename


def get_output_base_name(config):
    return config["name"]


def get_extended_output_path(config):
    return PROJECT_ROOT / f"{get_output_base_name(config)}_extended.csv"


def get_model_output_path(config):
    return SCRIPT_DIR / f"{get_output_base_name(config)}_stationary_gen.pth"


def get_plot_output_path(config):
    return SCRIPT_DIR / f"{get_output_base_name(config)}_stationary_path.png"


def get_report_dir(config):
    return REPORTS_ROOT / get_output_base_name(config)


def _last_valid_value(series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[-1]


def _shift_weekend_target_rows(raw_df, target_col):
    shifted_df = raw_df.copy()
    weekend_target_mask = shifted_df[target_col].notna() & (shifted_df["Date"].dt.dayofweek >= 5)
    shifted_df.loc[weekend_target_mask, "Date"] = shifted_df.loc[weekend_target_mask, "Date"].map(
        lambda dt: (dt - BDay(1)).normalize()
    )
    return shifted_df, int(weekend_target_mask.sum())


def _prepare_gold_gepu_business_interpolate(config):
    source_path = PROJECT_ROOT / config["source_file"]
    prepared_path = PROJECT_ROOT / config["prepared_file"]

    raw_df = pd.read_csv(source_path)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"], errors="coerce")
    raw_df = raw_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    value_cols = [column for column in raw_df.columns if column != "Date"]
    for column in value_cols:
        raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")

    raw_df, shifted_weekend_rows = _shift_weekend_target_rows(raw_df, config["target_col"])
    collapsed_df = raw_df.groupby("Date", as_index=False).agg({column: _last_valid_value for column in value_cols})

    valid_target_dates = collapsed_df.loc[collapsed_df[config["target_col"]].notna(), "Date"]
    if valid_target_dates.empty:
        raise ValueError(f"{config['source_file']} does not contain any non-null {config['target_col']} values")

    first_target_date = valid_target_dates.min()
    last_target_date = valid_target_dates.max()
    business_index = pd.date_range(first_target_date, last_target_date, freq="B")

    aligned_df = pd.DataFrame(index=business_index)
    aligned_df.index.name = "Date"
    aligned_df = aligned_df.join(collapsed_df.set_index("Date")[value_cols], how="left")
    missing_before_fill = aligned_df.isna().sum()

    filled_df = aligned_df.interpolate(method="time", limit_direction="both").ffill().bfill()
    missing_after_fill = filled_df.isna().sum()
    filled_df = filled_df.reset_index()
    filled_df.to_csv(prepared_path, index=False)

    summary = {
        "asset": config["asset"],
        "source_file": config["source_file"],
        "prepared_file": config["prepared_file"],
        "rows_raw": int(len(raw_df)),
        "rows_after_date_collapse": int(len(collapsed_df)),
        "rows_prepared": int(len(filled_df)),
        "first_target_date": str(first_target_date.date()),
        "last_target_date": str(last_target_date.date()),
        "weekend_target_rows_shifted_to_prior_business_day": shifted_weekend_rows,
        "raw_missing_pct": {
            column: round(float(raw_df[column].isna().mean() * 100.0), 2)
            for column in value_cols
        },
        "prepared_missing_cells_before_fill": {column: int(missing_before_fill[column]) for column in value_cols},
        "prepared_missing_cells_after_fill": {column: int(missing_after_fill[column]) for column in value_cols},
        "notes": [
            "Prepared on a business-day calendar from the first to last observed target date.",
            "Weekend rows carrying Gold_Futures were shifted to the prior business day before collapsing duplicates.",
            "Numeric columns were filled with time interpolation followed by forward/back fill.",
        ],
    }
    return prepared_path, summary


def ensure_prepared_source(config):
    preprocess_mode = config.get("preprocess")
    if preprocess_mode is None:
        return PROJECT_ROOT / config["source_file"], None
    if preprocess_mode == "gold_gepu_business_interpolate":
        prepared_path, summary = _prepare_gold_gepu_business_interpolate(config)
    else:
        raise ValueError(f"Unsupported preprocess mode: {preprocess_mode}")

    report_dir = get_report_dir(config)
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "source_preparation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return prepared_path, summary
