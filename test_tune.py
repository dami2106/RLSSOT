import subprocess
import os

# Activate the conda environment
conda_env_name = "SOTA"
os.system(f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate {conda_env_name}")

# Hyperparameters as variables
hyperparams = {
    "alpha_train": 0.3,            # Weighting of KOT term on frame features in OT during training
    "alpha_eval": 0.6,             # Weighting of KOT term on frame features in OT during evaluation
    "lambda_frames_train": 0.05,   # Penalty for balanced frame assumption during training
    "lambda_actions_train": 0.11,  # Penalty for balanced action assumption during training
    "lambda_frames_eval": 0.05,    # Penalty for balanced frame assumption during evaluation
    "lambda_actions_eval": 0.01,   # Penalty for balanced action assumption during evaluation
    "eps_train": 0.07,             # Entropy regularization for OT during training
    "eps_eval": 0.04,              # Entropy regularization for OT during evaluation
    "radius_gw": 0.04,             # Radius parameter for GW structure loss
    "rho": 0.15,                   # Global structure weighting factor
    "n_frames": 3,                 # Number of frames sampled per video for train/val
    "learning_rate": 1e-1,         # Learning rate for the optimizer
    "weight_decay": 1e-3,          # Weight decay for the optimizer
    "use_std_feats": True          # Enable standardization of features
}

# Translate boolean to flag
std_feats_flag = "--std-feats" if hyperparams["use_std_feats"] else ""

# Command for the training script
command = [
    "python3", "train.py",
    "--activity", "all",
    "--dataset", "desktop_assembly",
    "--group", "sweep_baseline_7",
    "--n-epochs", "30",
    "--visualize",
    "--n-clusters", "3",
    "--val-freq", "5",
    "--gpu", "0",
    "--seed", "0",
    "--batch-size", "2",
    "--wandb",
    "--learning-rate", str(hyperparams["learning_rate"]),
    "--weight-decay", str(hyperparams["weight_decay"]),
    "--layers", "11", "11", "11",
    "--n-ot-train", "25", "1",
    "--n-ot-eval", "25", "1",
    "--alpha-train", str(hyperparams["alpha_train"]),
    "--alpha-eval", str(hyperparams["alpha_eval"]),
    "--lambda-frames-train", str(hyperparams["lambda_frames_train"]),
    "--lambda-actions-train", str(hyperparams["lambda_actions_train"]),
    "--lambda-frames-eval", str(hyperparams["lambda_frames_eval"]),
    "--lambda-actions-eval", str(hyperparams["lambda_actions_eval"]),
    "--eps-train", str(hyperparams["eps_train"]),
    "--eps-eval", str(hyperparams["eps_eval"]),
    "--radius-gw", str(hyperparams["radius_gw"]),
    "--rho", str(hyperparams["rho"]),
    std_feats_flag,
    "--n-frames", str(hyperparams["n_frames"])
]

# Remove empty flags
command = [arg for arg in command if arg]

# Run the command
subprocess.run(command)
