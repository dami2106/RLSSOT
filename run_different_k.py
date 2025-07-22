import pandas as pd
import subprocess
import csv
import re

TASK_NAME = "wsws_static"
DATASET_SIZE = "pixels_big"
CLUSTER_SIZES = [3]  # Example list, change as needed

# Load and sort CSV
df = pd.read_csv(f'Traces/{TASK_NAME}/{TASK_NAME}_{DATASET_SIZE}/best.csv')

def build_cli(row, cluster_size):
    args = [
        f"--alpha-eval {row['params_alpha-eval']}",
        f"--alpha-train {row['params_alpha-train']}",
        f"--batch-size {row['params_batch-size']}",
        f"--eps-eval {row['params_eps-eval']}",
        f"--eps-train {row['params_eps-train']}",
        f"--lambda-actions-eval {row['params_lambda-actions-eval']}",
        f"--lambda-actions-train {row['params_lambda-actions-train']}",
        f"--lambda-frames-eval {row['params_lambda-frames-eval']}",
        f"--lambda-frames-train {row['params_lambda-frames-train']}",
        f"--learning-rate {row['params_learning-rate']}",
        f"--n-epochs {row['params_n-epochs']}",
        f"--n-frames {row['params_n-frames']}",
        f"--radius-gw {row['params_radius-gw']}",
        f"--rho {row['params_rho']}",
        f"--weight-decay {row['params_weight-decay']}",
        f"--dataset {TASK_NAME}/{TASK_NAME}_{DATASET_SIZE}",
        "--feature-name pca_features",
        "--save-directory runs",
        f"--run {TASK_NAME}_{DATASET_SIZE}_{row['number']}_k{cluster_size}",
        "--val-freq 5",
        "--layers 650 300 40",
        "--seed 0",
        "--visualize",
        "--log",
        f"--n-clusters {cluster_size}",
    ]
    if row['params_ub-frames']: args.append("--ub-frames")
    if row['params_ub-actions']: args.append("--ub-actions")
    if row['params_std-feats']: args.append("--std-feats")
    return " ".join(args)

pattern = r"\b(test_\w+)\b[^\d\-]*([0-9]*\.?[0-9]+)"

results = []

for cluster_k in CLUSTER_SIZES:
    row = df.iloc[0]  # Use the best row (or change as needed)
    cli = build_cli(row, cluster_k)
    print(f"Running with cluster_k={cluster_k}")
    result = subprocess.run(f"python src/train.py {cli}", shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running command for cluster_k={cluster_k}: {result.stderr}")
        continue


    matches = re.findall(pattern, result.stdout)
    metrics = {metric: float(value) for metric, value in matches}
    results.append({
        "cluster_k": cluster_k,
        "f1_per": metrics.get("test_f1_per", 0),
        "f1_full": metrics.get("test_f1_full", 0),
        "miou_per": metrics.get("test_miou_per", 0),
        "miou_full": metrics.get("test_miou_full", 0),
        "mof_per": metrics.get("test_mof_per", 0),
        "mof_full": metrics.get("test_mof_full", 0),
    })

out_csv = f"{TASK_NAME}_{DATASET_SIZE}_{'_'.join(map(str, CLUSTER_SIZES))}.csv"
df_out = pd.DataFrame(results)
df_out.to_csv(out_csv, index=False)
print(f"Results saved to {out_csv}")
