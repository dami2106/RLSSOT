import os
import numpy as np
import argparse
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.metrics import confusion_matrix
import joblib
from skill_helpers import *

dir_ = 'Data/stone_pick_random_pixels_big'
files = os.listdir(dir_ + '/groundTruth')

# argparse: choose whether to train models for 'start' or 'end' detection
parser = argparse.ArgumentParser(description="Train One-Class SVM models to detect 'start' or 'end' states for each skill.")
parser.add_argument(
    "--phase",
    choices=["start", "end"],
    help="Which type of states to model as positive examples: 'start' or 'end'"
)
args = parser.parse_args()


nu_grid = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
gamma_grid = ['scale', 'auto', 1e-3, 1e-2, 1e-1]

rng = np.random.default_rng(42)
results = {}
skills = get_unique_skills(files)
skill_data = {}
for skill in skills:
    start_states, end_states, other_states = get_start_end_states(skill, files)
    skill_data[skill] = {
        'start_states': start_states,
        'end_states': end_states,
        'other_states': other_states
    }

    # Choose positives and negatives based on requested phase
    if args.phase == "start":
        X_pos_all = start_states
        X_neg_all = np.vstack([arr for arr in [end_states, other_states]]) if len(end_states) and len(other_states) else (
            end_states if len(end_states) else other_states
        )
        model_dir = f"{dir_}/start_models"
        pos_name = "start_states"
    else:  # args.phase == "end"
        X_pos_all = end_states
        X_neg_all = np.vstack([arr for arr in [start_states, other_states]]) if len(start_states) and len(other_states) else (
            start_states if len(start_states) else other_states
        )
        model_dir = f"{dir_}/end_models"
        pos_name = "end_states"


    X_pos_train, X_pos_val = train_test_split(X_pos_all, test_size=0.05, random_state=42, shuffle=True)

    n_val_neg = min(len(X_pos_val), X_neg_all.shape[0])
    idx = rng.choice(X_neg_all.shape[0], size=n_val_neg, replace=False)
    X_neg_val = X_neg_all[idx]
    
    best = {"f1": -1.0, "nu": None, "gamma": None, "clf": None}
    for nu in nu_grid:
        for gamma in gamma_grid:
            try:
                f1, clf = evaluate_ocsvm(X_pos_train, X_pos_val, X_neg_val, nu=nu, gamma=gamma)
            except Exception as e:
                continue
            if f1 > best["f1"]:
                best.update({"f1": f1, "nu": nu, "gamma": gamma, "clf": clf})

    final_clf = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                              OneClassSVM(kernel="rbf", nu=best["nu"], gamma=best["gamma"]))
    final_clf.fit(X_pos_all)

    n_test = min(len(X_pos_all), X_neg_all.shape[0])

    # all positives
    y_pos_true = np.ones(len(X_pos_all), dtype=int)
    y_pos_pred = (final_clf.predict(X_pos_all) == 1).astype(int)

    # same number of negatives
    neg_idx = rng.choice(X_neg_all.shape[0], size=n_test, replace=False)
    X_neg_test = X_neg_all[neg_idx]
    y_neg_true = np.zeros(n_test, dtype=int)
    y_neg_pred = (final_clf.predict(X_neg_test) == 1).astype(int)

    # combine
    y_true = np.hstack([y_pos_true, y_neg_true])
    y_pred = np.hstack([y_pos_pred, y_neg_pred])

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n=== Skill: {skill} ({args.phase}) ===")
    print(f"Best params -> nu={best['nu']}, gamma={best['gamma']}, val_F1={best['f1']:.3f}")
    print(f"Test metrics (pos={pos_name}, neg=sampled equally):")
    print(f"  Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print("Confusion Matrix [rows=true {neg,pos}, cols=pred {neg,pos}]:")
    print(cm)
    print(classification_report(y_true, y_pred, target_names=["neg", "pos"], zero_division=0))
    print("="*50)
    # Save the final model retrained on all positives to the appropriate directory
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(final_clf, f"{model_dir}/{skill}_best_model.joblib")
