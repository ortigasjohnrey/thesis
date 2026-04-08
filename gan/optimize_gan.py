import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import optuna
from optuna.samplers import TPESampler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "gan_validation"
OUTPUT_DIR = REPORTS_ROOT / "optuna"
GENERATED_PATHS = [
    PROJECT_ROOT / "gold_RRL_interpolate_extended.csv",
    PROJECT_ROOT / "silver_RRL_interpolate_extended.csv",
    SCRIPT_DIR / "gold_RRL_interpolate_stationary_gen.pth",
    SCRIPT_DIR / "silver_RRL_interpolate_stationary_gen.pth",
    SCRIPT_DIR / "gold_RRL_interpolate_stationary_path.png",
    SCRIPT_DIR / "silver_RRL_interpolate_stationary_path.png",
    SCRIPT_DIR / "gan_training_stationary.log",
    REPORTS_ROOT / "gold_RRL_interpolate",
    REPORTS_ROOT / "silver_RRL_interpolate",
]

TUNING_SEED = int(os.getenv("GAN_OPTUNA_SEED", "42"))
N_TRIALS = int(os.getenv("GAN_OPTUNA_TRIALS", "12"))
TRIAL_EPOCHS = int(os.getenv("GAN_OPTUNA_EPOCHS", "40"))
GOLD_SEEDS = os.getenv("GAN_OPTUNA_GOLD_SEEDS", "0,1,2")
SILVER_SEEDS = os.getenv("GAN_OPTUNA_SILVER_SEEDS", "0")
NUM_CANDIDATES = os.getenv("GAN_OPTUNA_NUM_CANDIDATES", "12")


def clean_trial_outputs():
    for path in GENERATED_PATHS:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def run_script(script_name: str, extra_env: dict):
    env = os.environ.copy()
    env.update(extra_env)
    cmd = [sys.executable, str(SCRIPT_DIR / script_name)]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").lower()
        if "out of memory" in message or "cuda out of memory" in message:
            raise RuntimeError("oom")
        raise RuntimeError(completed.stderr or completed.stdout or f"{script_name} failed")


def load_selected_metrics(dataset_name: str):
    path = REPORTS_ROOT / dataset_name / "selected_candidate_metrics.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def objective(trial: optuna.Trial):
    clean_trial_outputs()

    params = {
        "GAN_EPOCHS": str(TRIAL_EPOCHS),
        "GAN_BATCH_SIZE": str(trial.suggest_categorical("batch_size", [256, 384, 512])),
        "GAN_LR_G": str(trial.suggest_float("lr_g", 1e-4, 5e-4, log=True)),
        "GAN_LR_D": str(trial.suggest_float("lr_d", 1e-4, 5e-4, log=True)),
        "GAN_N_CRITIC": str(trial.suggest_int("n_critic", 2, 4)),
        "GAN_GEN_HIDDEN_SIZE": str(trial.suggest_categorical("gen_hidden", [96, 128, 160, 192])),
        "GAN_DISC_HIDDEN_SIZE": str(trial.suggest_categorical("disc_hidden", [96, 128, 160, 192])),
        "GAN_GEN_DROPOUT": str(trial.suggest_float("gen_dropout", 0.05, 0.20)),
        "GAN_MOMENT_LOSS_WEIGHT": str(trial.suggest_float("moment_loss_weight", 0.4, 1.2)),
        "GAN_DRIFT_LOSS_WEIGHT": str(trial.suggest_float("drift_loss_weight", 0.1, 0.6)),
        "GAN_NUM_CANDIDATES": NUM_CANDIDATES,
        "GAN_GOLD_SEEDS": GOLD_SEEDS,
        "GAN_SILVER_SEEDS": SILVER_SEEDS,
        "GAN_STRICT_READINESS": "0",
    }

    try:
        run_script("generate_gan_data.py", params)
        run_script("report_gan_validity.py", params)
    except RuntimeError as exc:
        if str(exc) == "oom":
            raise optuna.TrialPruned("CUDA OOM")
        raise

    gold = load_selected_metrics("gold_RRL_interpolate")
    silver = load_selected_metrics("silver_RRL_interpolate")

    gold_score = gold["quality_score"]
    silver_score = silver["quality_score"]
    objective_value = 0.7 * gold_score + 0.3 * silver_score

    trial.set_user_attr("gold_quality_label", gold["quality_label"])
    trial.set_user_attr("silver_quality_label", silver["quality_label"])
    trial.set_user_attr("gold_avg_ks_stat", gold["avg_ks_stat"])
    trial.set_user_attr("silver_avg_ks_stat", silver["avg_ks_stat"])

    return objective_value


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sampler = TPESampler(seed=TUNING_SEED)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="gan_quality_optuna",
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params_path = OUTPUT_DIR / "best_gan_params.json"
    trial_report_path = OUTPUT_DIR / "gan_optuna_trials.csv"

    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_value": study.best_value,
                "best_params": study.best_trial.params,
                "best_user_attrs": study.best_trial.user_attrs,
            },
            f,
            indent=2,
        )

    rows = []
    for trial in study.trials:
        rows.append(
            {
                "trial_number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "params": json.dumps(trial.params),
                "user_attrs": json.dumps(trial.user_attrs),
            }
        )

    import pandas as pd

    pd.DataFrame(rows).to_csv(trial_report_path, index=False)
    print(f"Best GAN objective: {study.best_value}")
    print(f"Best params saved to: {best_params_path}")
    print(f"Trial report saved to: {trial_report_path}")


if __name__ == "__main__":
    main()
