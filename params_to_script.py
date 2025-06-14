import os
import sys
# ROW = ("pixels_big	1128	0.6056473136	2025-04-13 8:38:17	2025-04-13 8:38:36	0 days 00:00:18.909201	0.8230799961	0.07434594466	16	0.01340313487	0.005432281575	0.03039409858	0.0825285425	0.04217558672	0.08423682501	0.0001105710232	18	36	0.01458080174	0.02494160656	FALSE	TRUE	TRUE	0.0001344258452	COMPLETE	")

#Take row in as argument
ROW = (str(sys.argv[1]))

TASK_NAME = "wsws_static"
N_CLUSTERS = 2


TASK_TYPE = ROW.split("\t")[0]
ACTIONS = 'actions' in TASK_TYPE

if ACTIONS:
    TASK_TYPE = TASK_TYPE.replace('_actions', '')

if not ACTIONS:
    if TASK_TYPE == "pixels":
        LAYERS = "300 150 40"
        FEATURE_NAME = "pca_features"
    elif TASK_TYPE == "pixels_big":
        LAYERS = "650 300 40"
        FEATURE_NAME = "pca_features"
    elif TASK_TYPE in ("symbolic", "symbolic_big"):
        LAYERS = "1087 512 50"
        FEATURE_NAME = "symbolic_obs"
    else:
        raise ValueError(f"Unknown TASK_TYPE: {TASK_TYPE}")
else:
    if TASK_TYPE == "pixels":
        LAYERS = "317 150 40"
        FEATURE_NAME = "pca_features_with_actions"
    elif TASK_TYPE == "pixels_big":
        LAYERS = "667 300 40"
        FEATURE_NAME = "pca_features_with_actions"
    elif TASK_TYPE in ("symbolic", "symbolic_big"):
        LAYERS = "1104 512 50"
        FEATURE_NAME = "symbolic_obs_with_actions"
    else:
        raise ValueError(f"Unknown TASK_TYPE: {TASK_TYPE}")

# Dataset parameters (not in the Excel row)
DATASET             = f"{TASK_NAME}/{TASK_NAME}_{TASK_TYPE}"
SAVE_DIRECTORY      = "runs"
RUN_NAME            = f"{TASK_NAME}_{TASK_TYPE}"
VAL_FREQ            = 5

# General parameters (not in the Excel row)
LOG                 = True
VISUALIZE           = True
SEED                = 0
TRAIN_SCRIPT_PATH   = "src/train.py"

if not ACTIONS:
    SCRIPT_FILENAME     = f"{TASK_NAME}_{TASK_TYPE}.sh"
else:
    SCRIPT_FILENAME     = f"{TASK_NAME}_{TASK_TYPE}_actions.sh"


def row_to_script_and_save(row_str: str, delimiter: str = "\t") -> None:
    """
    Parse one Excel row (delimiter-separated string) and
    generate a bash script file named by SCRIPT_FILENAME.
    Expects exactly 25 fields in this order:
      Task Type, number, value, datetime_start, datetime_complete, duration,
      params_alpha-eval, params_alpha-train, params_batch-size,
      params_eps-eval, params_eps-train,
      params_lambda-actions-eval, params_lambda-actions-train,
      params_lambda-frames-eval, params_lambda-frames-train,
      params_learning-rate, params_n-epochs, params_n-frames,
      params_radius-gw, params_rho,
      params_std-feats, params_ub-actions, params_ub-frames,
      params_weight-decay, state
    """
    parts = row_str.strip().split(delimiter)
    if len(parts) != 25:
        raise ValueError(f"Expected 25 fields, got {len(parts)}")

    (
        task_type, number, value, datetime_start, datetime_complete, duration,
        alpha_eval, alpha_train, batch_size, eps_eval, eps_train,
        lambda_actions_eval, lambda_actions_train,
        lambda_frames_eval, lambda_frames_train,
        learning_rate, n_epochs, n_frames,
        radius_gw, rho,
        std_feats, ub_actions, ub_frames,
        weight_decay, state
    ) = parts

    def to_bool(s: str) -> bool:
        return s.strip().upper() == "TRUE"

    std_feats   = to_bool(std_feats)
    ub_actions  = to_bool(ub_actions)
    ub_frames   = to_bool(ub_frames)

    # --- build each section of the script ---
    hyper = f"""# --- Hyperparameters and OT segmentation parameters ---
ALPHA_TRAIN={alpha_train}
ALPHA_EVAL={alpha_eval}

UB_FRAMES={str(ub_frames).lower()}
UB_ACTIONS={str(ub_actions).lower()}

LAMBDA_FRAMES_TRAIN={lambda_frames_train}
LAMBDA_ACTIONS_TRAIN={lambda_actions_train}
LAMBDA_FRAMES_EVAL={lambda_frames_eval}
LAMBDA_ACTIONS_EVAL={lambda_actions_eval}

EPS_TRAIN={eps_train}
EPS_EVAL={eps_eval}
RADIUS_GW={radius_gw}
"""

    dataset = f"""# --- Dataset parameters ---
DATASET="{DATASET}"
FEATURE_NAME="{FEATURE_NAME}"
STD_FEATS={str(std_feats).lower()}
SAVE_DIRECTORY="{SAVE_DIRECTORY}"
RUN="{RUN_NAME}"
VAL_FREQ={VAL_FREQ}
"""

    general = f"""# --- General parameters ---
N_EPOCHS={n_epochs}
BATCH_SIZE={batch_size}
N_FRAMES={n_frames}
LEARNING_RATE={learning_rate}
WEIGHT_DECAY={weight_decay}
LOG={str(LOG).lower()}
VISUALIZE={str(VISUALIZE).lower()}
SEED={SEED}
RHO={rho}
N_CLUSTERS={N_CLUSTERS}
LAYERS="{LAYERS}"
"""

    # --- Build the CMD line ---
    flags = [
        "--alpha-train $ALPHA_TRAIN",
        "--alpha-eval  $ALPHA_EVAL",
        "--lambda-frames-train $LAMBDA_FRAMES_TRAIN",
        "--lambda-actions-train $LAMBDA_ACTIONS_TRAIN",
        "--lambda-frames-eval  $LAMBDA_FRAMES_EVAL",
        "--lambda-actions-eval  $LAMBDA_ACTIONS_EVAL",
        "--eps-train   $EPS_TRAIN",
        "--eps-eval    $EPS_EVAL",
        "--radius-gw   $RADIUS_GW",
        "--dataset     $DATASET",
        "--n-frames    $N_FRAMES",
        "--save-directory $SAVE_DIRECTORY",
        "--n-epochs    $N_EPOCHS",
        "--batch-size  $BATCH_SIZE",
        "--learning-rate $LEARNING_RATE",
        "--weight-decay $WEIGHT_DECAY",
        "--layers      $LAYERS",
        "--rho         $RHO",
        "--n-clusters  $N_CLUSTERS",
        "--val-freq    $VAL_FREQ",
        "--seed        $SEED",
        "--run         $RUN",
        "--feature-name $FEATURE_NAME",
    ]
    cmd_body = " \\\n  ".join(flags)
    cmd = f'CMD="python {TRAIN_SCRIPT_PATH} \\\n  {cmd_body}"\n'

    # Append boolean flags
    if ub_frames:
        cmd += 'CMD="$CMD --ub-frames"\n'
    if ub_actions:
        cmd += 'CMD="$CMD --ub-actions"\n'
    if std_feats:
        cmd += 'CMD="$CMD --std-feats"\n'
    if VISUALIZE:
        cmd += 'CMD="$CMD --visualize"\n'
    if LOG:
        cmd += 'CMD="$CMD --log"\n'

    # Final glue
    script = f"""#!/bin/bash

{hyper}
{dataset}
{general}

# --- Build the command ---
{cmd}
# --- Execute the command ---
echo "Running command:"
echo "$CMD"
eval $CMD
"""

    # Write to file
    with open(f'Scripts/{TASK_NAME}/' + SCRIPT_FILENAME, "w") as f:
        f.write(script)
    # Make it executable
    os.chmod(f'Scripts/{TASK_NAME}/' + SCRIPT_FILENAME, 0o755)



row_to_script_and_save(ROW)
print(f"Script written to {SCRIPT_FILENAME}")
