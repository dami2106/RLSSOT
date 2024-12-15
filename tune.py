import itertools
import subprocess
import os
import csv
import re
from tqdm import tqdm
import random
import string

conda_env_name = "SOTA"
os.system(f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate {conda_env_name}")


# Define ranges for hyperparameters to test
ALPHA_VALUES = [0.1, 0.3, 0.6, 1.0, 2.0]  # Example values for ALPHA_TRAIN and ALPHA_EVAL
LAMBDA_VALUES = [0.01, 0.05, 0.1, 0.5, 0.8, 1.0, 2.0]  # Example values for LAMBDA_*_TRAIN and LAMBDA_*_EVAL
EPS_VALUES = [0.01, 0.03, 0.06, 0.1, 1.0]  # Example values for EPS_TRAIN and EPS_EVAL
CLUSTERS = [3, 4]

def generate_random_string(length=15):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


# Base command
base_command = [
    "python3", "train.py",
    "--activity", "all",
    "--dataset", "desktop_assembly",
    "--group", generate_random_string(),
    "--n-epochs", "30",
    "--visualize",
    "--val-freq", "5",
    "--gpu", "1",
    "--seed", "0",
    "--batch-size", "2",
    "--wandb",
    "--learning-rate", "1e-1",
    "--weight-decay", "1e-3",
    "--layers", "11", "11", "11",
    "--n-ot-train", "25", "1",
    "--n-ot-eval", "25", "1",
    "--radius-gw", "0.04",
    "--rho", "0.15",
    "--n-frames", "3"
]

base_command.append("--std-feats")

# # Create all combinations of hyperparameters
combinations = itertools.product(ALPHA_VALUES, ALPHA_VALUES, \
                                LAMBDA_VALUES, LAMBDA_VALUES, LAMBDA_VALUES, LAMBDA_VALUES,\
                                EPS_VALUES, EPS_VALUES, CLUSTERS)

# Directory to save logs
os.makedirs("logs", exist_ok=True)

# CSV file to save configurations
csv_file = "logs/hyperparameter_results.csv"

# CSV headers
csv_headers = [
    "run_id", "alpha-train", "alpha-eval", "lambda-frames-train", "lambda-actions-train",
    "lambda-frames-eval", "lambda-actions-eval", "eps-train", "eps-eval",
    "radius-gw", "rho", "n-frames", "n-clusters", "learning-rate",
    "ub-frames", "ub-actions", "weight-decay", "Data", "Epochs", "MLP", "STD-Feats",
    "State", "Actions", "Experiment", "F1 Full", "F1 Per", "MIOU Full", "MIOU Per", "MOF Full", "MOF Per"
]

# Initialize CSV file
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

# Iterate over combinations and run the script

run_id = 0

for alpha_t, alpha_e, lambda_f_t, lambda_a_t, lambda_f_e, lambda_a_e, eps_t, eps_e, n_clusters in tqdm(combinations):
    # Add hyperparameters to the command
    command = base_command + [
        "--alpha-train", str(alpha_t),
        "--alpha-eval", str(alpha_e),
        "--lambda-frames-train", str(lambda_f_t),
        "--lambda-actions-train", str(lambda_a_t),
        "--lambda-frames-eval", str(lambda_f_e),
        "--lambda-actions-eval", str(lambda_a_e),
        "--eps-train", str(eps_t),
        "--eps-eval", str(eps_e),
        "--n-clusters", str(n_clusters),
    ]
    
    # Log file for the current combination
    log_file = f"logs/run_{run_id}.log"
    
    print(f"Running:", alpha_t, alpha_e, lambda_f_t, lambda_a_t, lambda_f_e, lambda_a_e, eps_t, eps_e)
    
    command = [arg for arg in command if arg]

    # Run the script and capture the output
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    pattern = r"(\btest_\w+\b)\s+([\d\.]+)"
    matches = re.findall(pattern, process.stdout)
    metrics = {metric: float(value) for metric, value in matches}


    # Save the configuration and results in the CSV
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        
        # Extract metrics/results from the output (example placeholders)
        model_data = {
            "run_id": run_id,
            "alpha-train": alpha_t,
            "alpha-eval": alpha_e,
            "lambda-frames-train": lambda_f_t,
            "lambda-actions-train": lambda_a_t,
            "lambda-frames-eval": lambda_f_e,
            "lambda-actions-eval": lambda_a_e,
            "eps-train": eps_t,
            "eps-eval": eps_e,
            "radius-gw": 0.04,  # Fixed value
            "rho": 0.15,  # Fixed value
            "n-frames": 3,  # Fixed value
            "n-clusters": n_clusters,  # Fixed value
            "learning-rate": 1e-1,  # Fixed value
            "ub-frames": "FALSE",  # Placeholder
            "ub-actions": "FALSE",  # Placeholder
            "weight-decay": 1e-3,  # Fixed value
            "Data": "100",  # Fixed value
            "Epochs": 30,  # Fixed value
            "MLP": "11 11 11",  # MLP
            "STD-feats": "TRUE",  # Fixed value
            "State" : "Simple",
            "Actions" : "FALSE",
            "Experiment" : "Standard Env;Random All",
            "F1 Full": metrics["test_f1_full"],
            "F1 Per": metrics["test_f1_per"],
            "MIOU Full": metrics["test_miou_full"],
            "MIOU Per": metrics["test_miou_per"],
            "MOF Full": metrics["test_mof_full"],
            "MOF Per": metrics["test_mof_per"]
        }
        
        # Write row
        writer.writerow(model_data.values())

    # Log output
    with open(log_file, "w") as log:
        log.write(process.stdout)
        log.write(process.stderr)
    
    run_id += 1

print("Hyperparameter tuning complete. Check 'hyperparameter_results.csv' and 'logs' directory for details.")
