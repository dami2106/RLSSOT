import optuna
import subprocess
import json
import re
import pandas as pd
import joblib
import os
import argparse

def save_study_progress(study, save_dir):
    """Saves the study results to a CSV file and the study object to a pickle file."""
    df = study.trials_dataframe()
    df.to_csv(save_dir + '/optuna_results.csv', index=False)
    joblib.dump(study, args.save + '/optuna_study.pkl')

def objective(trial, dataset):
    """Objective function to optimize hyperparameters"""
    # Define hyperparameter search space
    params = {
        "alpha-train": trial.suggest_float("alpha-train", 0.01, 1),
        "alpha-eval": trial.suggest_float("alpha-eval", 0.01, 1),
        "lambda-frames-train": trial.suggest_float("lambda-frames-train", 0.001, 0.1),
        "lambda-actions-train": trial.suggest_float("lambda-actions-train", 0.001, 0.1),
        "lambda-frames-eval": trial.suggest_float("lambda-frames-eval", 0.001, 0.1),
        "lambda-actions-eval": trial.suggest_float("lambda-actions-eval", 0.001, 0.1),
        "eps-train": trial.suggest_float("eps-train", 0.0001, 0.3),
        "eps-eval": trial.suggest_float("eps-eval", 0.0001, 0.3),
        "radius-gw": trial.suggest_float("radius-gw", 0.001, 0.1),
        "learning-rate": trial.suggest_float("learning-rate", 1e-5, 1e-1, log=True),
        "weight-decay": trial.suggest_float("weight-decay", 1e-8, 1e-3, log=True),
        "batch-size": trial.suggest_categorical("batch-size", [8, 16, 32]),
        "n-epochs": trial.suggest_int("n-epochs", 5, 30),
        "ub-frames": trial.suggest_categorical("ub-frames", [True, False]),
        "ub-actions": trial.suggest_categorical("ub-actions", [True, False]),
        "std-feats": trial.suggest_categorical("std-feats", [True, False]),
        "rho": trial.suggest_float("rho", 0.001, 0.3),
        "n-frames": trial.suggest_int("n-frames", 5, 80),
    }

    # Build command line arguments from hyperparameters
    cli_args = " ".join(
        [f"--{k} {v}" for k, v in params.items() if not isinstance(v, bool)]
    )
    if params["ub-frames"]:
        cli_args += " --ub-frames"
    if params["ub-actions"]:
        cli_args += " --ub-actions"
    if params["std-feats"]:
        cli_args += " --std-feats"

    # Append dataset parameter to the command
    cli_args += f" --dataset {dataset}"

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
            (0.05 * test_f1_per + 0.05 * test_miou_per + 0.05 * test_mof_per)
    
    return score  # Higher is better

# Run optimization

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume study')
    parser.add_argument('--dataset', type=str, required=True, help='Path of the dataset to use')
    parser.add_argument('--directory', type=str, default='runs/test',
                        help='Path to save the study object (pickle file)')
    args = parser.parse_args()

    if args.resume:
        study = joblib.load(args.directory + '/optuna_study.pkl')
    else:
        study = optuna.create_study(direction="maximize")

    # Pass the dataset argument into the objective function via a lambda
    study.optimize(lambda trial: objective(trial, args.dataset), n_trials=1500,
                   callbacks=[lambda study, trial: save_study_progress(study, args.directory)])

    # Save final results and study object
    save_study_progress(study, args.directory)