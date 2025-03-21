import optuna
import subprocess
import json
import re
import pandas as pd
import joblib
import os

def save_study_progress(study, filename="optuna_study_results.csv"):
    """ Saves the study results to a CSV file after each trial. """
    df = study.trials_dataframe()
    df.to_csv(filename, index=False)

def objective(trial):
    """ Objective function to optimize hyperparameters """
    # Define hyperparameter search space
    params = {
        "alpha-train": trial.suggest_float("alpha-train", 0.2, 0.8),
        "alpha-eval": trial.suggest_float("alpha-eval", 0.2, 0.8),
        "lambda-frames-train": trial.suggest_float("lambda-frames-train", 0.001, 0.1),
        "lambda-actions-train": trial.suggest_float("lambda-actions-train", 0.001, 0.1),
        "lambda-frames-eval": trial.suggest_float("lambda-frames-eval", 0.001, 0.1),
        "lambda-actions-eval": trial.suggest_float("lambda-actions-eval", 0.001, 0.1),
        "eps-train": trial.suggest_float("eps-train", 0.001, 0.1),
        "eps-eval": trial.suggest_float("eps-eval", 0.001, 0.1),
        "radius-gw": trial.suggest_float("radius-gw", 0.001, 0.1),
        "learning-rate": trial.suggest_float("learning-rate", 1e-5, 1e-1, log=True),
        "weight-decay": trial.suggest_float("weight-decay", 1e-6, 1e-3, log=True),
        "batch-size": trial.suggest_categorical("batch-size", [8]),
        "n-epochs": trial.suggest_int("n-epochs", 5, 30),
        "ub-frames": trial.suggest_categorical("ub-frames", [True, False]),
        "ub-actions": trial.suggest_categorical("ub-actions", [True, False]),
        "rho": trial.suggest_float("rho", 0.01, 0.3),
        "n-frames": trial.suggest_int("n-frames", 4, 25),
    }

    cli_args = " ".join(
        [f"--{k} {v}" for k, v in params.items() if not isinstance(v, bool)]
    )
    if params["ub-frames"]:
        cli_args += " --ub-frames"
    if params["ub-actions"]:
        cli_args += " --ub-actions"

    # Run model training with the given parameters
    try:
        result = subprocess.run(
            f"python src/train.py {cli_args}",
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        pattern = r"(\btest_\w+\b)\s+([\d\.]+)"
        matches = re.findall(pattern, result.stdout)
        metrics = {metric: float(value) for metric, value in matches}
    except subprocess.CalledProcessError as e:
        print(f"Error running training: {e}")
        print(f"Standard Output:\n{e.stdout}")
        print(f"Standard Error:\n{e.stderr}")
        return float('-inf')  # If training fails, return a bad score
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Output received:\n{result.stdout}")
        return float('-inf')  # If JSON decoding fails, return a bad score

    # Extract metrics
    test_f1_full = metrics.get("test_f1_full", 0)
    test_f1_per = metrics.get("test_f1_per", 0)
    test_miou_full = metrics.get("test_miou_full", 0)
    test_miou_per = metrics.get("test_miou_per", 0)
    test_mof_full = metrics.get("test_mof_full", 0)
    test_mof_per = metrics.get("test_mof_per", 0)

    # Compute weighted objective
    score = (0.3 * test_f1_full + 0.3 * test_miou_full + 0.3 * test_mof_full) + \
            (0.1 * test_f1_per + 0.1 * test_miou_per + 0.1 * test_mof_per)
    
    return score  # Higher is better

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000, callbacks=[lambda study, trial: save_study_progress(study)])

# Save final results and study object
save_study_progress(study)
joblib.dump(study, 'optuna_study.pkl')
