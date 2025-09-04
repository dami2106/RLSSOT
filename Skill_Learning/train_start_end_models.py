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
import matplotlib.pyplot as plt
import seaborn as sns
from old_helpers import *

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


nu_grid = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1]
gamma_grid = ['scale', 'auto'] + list(10.0**np.arange(-6, 1))  # 1e-6 .. 1e0

rng = np.random.default_rng(42)
results = {}
skills = get_unique_skills(dir_, files)
skill_data = {}
for skill in skills:
    start_states, end_states, other_states, all_skill_states = get_start_end_states(dir_, skill, files)
    skill_data[skill] = {
        'start_states': start_states,
        'end_states': end_states,
        'other_states': other_states,
        'all_skill_states': all_skill_states
    }

    # Choose positives and negatives based on requested phase
    if args.phase == "start":
        X_pos_all = all_skill_states
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
    results[skill] = {
    "threshold": float("nan"),   # OC-SVM has no probability threshold; keep column for side-by-side compare
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "confusion_matrix": cm
        }
    

    print(f"\n=== Skill: {skill} ({args.phase}) ===")
    print(f"Best params -> nu={best['nu']}, gamma={best['gamma']}, val_F1={best['f1']:.3f}")
    print(f"Test metrics (pos={pos_name}, neg=sampled equally):")
    print(f"  Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print("Confusion Matrix [rows=true {neg,pos}, cols=pred {neg,pos}]:")
    print(cm)
    print(classification_report(y_true, y_pred, target_names=["neg", "pos"], zero_division=0))
    print("="*50)
    # Save confusion matrix plot with metrics
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
    plt.title(f"Confusion Matrix for {skill} ({args.phase})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    metrics_text = f"Precision: {prec:.3f}\nRecall: {rec:.3f}\nF1: {f1:.3f}\nnu: {best['nu']}\ngamma: {best['gamma']}"
    plt.gcf().text(0.99, 0.01, metrics_text, fontsize=10, va='bottom', ha='right', bbox=dict(facecolor='white', alpha=0.7))
    img_filename = f"{skill}_{args.phase}.png"
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(img_filename, dpi=300)
    plt.close()
    # Save the final model retrained on all positives to the appropriate directory
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(final_clf, f"{model_dir}/{skill}_best_model.joblib")


# ========= Summary across skills (OC-SVM) =========
from math import isnan
import pandas as pd

rows = []
tot_tn = tot_fp = tot_fn = tot_tp = 0

for skill, res in results.items():
    cm = res["confusion_matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]
    tot_tn += int(tn); tot_fp += int(fp); tot_fn += int(fn); tot_tp += int(tp)

    support_pos = int(tp + fn)
    support_neg = int(tn + fp)
    support_all = support_pos + support_neg
    acc = (tp + tn) / support_all if support_all else float("nan")

    rows.append({
        "skill": skill,
        "pos_support": support_pos,
        "neg_support": support_neg,
        "threshold": res.get("threshold", float("nan")),  # keep same column as SVC
        "precision": res["precision"],
        "recall": res["recall"],
        "f1": res["f1"],
        "accuracy": acc,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    })

# Overall (micro) metrics
overall_support = tot_tp + tot_fp + tot_fn + tot_tn
overall_precision = (tot_tp / (tot_tp + tot_fp)) if (tot_tp + tot_fp) else 0.0
overall_recall    = (tot_tp / (tot_tp + tot_fn)) if (tot_tp + tot_fn) else 0.0
if (overall_precision + overall_recall) > 0:
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
else:
    overall_f1 = 0.0
overall_accuracy  = (tot_tp + tot_tn) / overall_support if overall_support else float("nan")

# Macro (mean across skills)
import numpy as np
macro_precision = float(np.mean([r["precision"] for r in rows])) if rows else float("nan")
macro_recall    = float(np.mean([r["recall"]    for r in rows])) if rows else float("nan")
macro_f1        = float(np.mean([r["f1"]        for r in rows])) if rows else float("nan")
macro_accuracy  = float(np.mean([r["accuracy"]  for r in rows])) if rows else float("nan")

# Pretty print (same style as SVC)
print("\n" + "="*80)
print("PER-SKILL METRICS (OC-SVM) — sorted by F1 desc")
print("="*80)

if pd is not None:
    df = pd.DataFrame(rows)
    df = df.sort_values("f1", ascending=False)
    disp_cols = ["skill", "pos_support", "neg_support", "threshold",
                 "precision", "recall", "f1", "accuracy", "tp", "fp", "fn"]
    for c in ["threshold", "precision", "recall", "f1", "accuracy"]:
        df[c] = df[c].astype(float).round(3)
    print(df[disp_cols].to_string(index=False))
else:
    rows_sorted = sorted(rows, key=lambda r: r["f1"], reverse=True)
    header = f"{'skill':18s} {'pos':>5s} {'neg':>5s} {'thr':>6s} {'P':>6s} {'R':>6s} {'F1':>6s} {'Acc':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s}"
    print(header)
    print("-"*len(header))
    for r in rows_sorted:
        thr = r['threshold']
        thr_print = f"{thr:.3f}" if isinstance(thr, float) and not isnan(thr) else "  nan"
        print(f"{r['skill']:18s} {r['pos_support']:5d} {r['neg_support']:5d} "
              f"{thr_print:>6s} {r['precision']:6.3f} {r['recall']:6.3f} "
              f"{r['f1']:6.3f} {r['accuracy']:6.3f} {r['tp']:5d} {r['fp']:5d} {r['fn']:5d}")

print("\n" + "="*80)
print("OVERALL (MICRO) METRICS — pooled over all skills (OC-SVM)")
print("="*80)
print(f"Support (all skills): {overall_support}")
print(f"TP={tot_tp}  FP={tot_fp}  FN={tot_fn}  TN={tot_tn}")
print(f"Precision: {overall_precision:.3f}  Recall: {overall_recall:.3f}  F1: {overall_f1:.3f}  Accuracy: {overall_accuracy:.3f}")

print("\n" + "="*80)
print("MACRO AVERAGES — mean of per-skill metrics (OC-SVM)")
print("="*80)
print(f"Precision: {macro_precision:.3f}  Recall: {macro_recall:.3f}  F1: {macro_f1:.3f}  Accuracy: {macro_accuracy:.3f}")
print("="*80 + "\n")