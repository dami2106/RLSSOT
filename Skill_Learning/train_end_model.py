import os
import json
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from skill_helpers import *
from joblib import dump
import pandas as pd 
from math import isnan

SEED = 42
rng = np.random.default_rng(SEED)

dir_ = 'Craftax-Skill-Data/Traces/stone_pickaxe_easy'

# Directory to save trained end models & metadata
models_dir = os.path.join(dir_, 'end_models')
os.makedirs(models_dir, exist_ok=True)
files = os.listdir(os.path.join(dir_, 'groundTruth'))


def make_clf(C=1.0, seed=SEED):
    base = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", dual="auto", max_iter=50000, C=C, random_state=seed)
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=cv)

def best_threshold_from_pr(y_true, p_scores):
    prec, rec, thr = precision_recall_curve(y_true, p_scores) 
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1s))
    if best_idx == 0:
        best_thr = thr[0] if len(thr) else 0.5
    elif best_idx - 1 < len(thr):
        best_thr = thr[best_idx - 1]
    else:
        best_thr = thr[-1] if len(thr) else 0.5
    return float(best_thr), float(f1s[best_idx])

def fit_with_threshold(X, y, C=1.0, seed=SEED):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, random_state=seed, stratify=y
    )
    clf_inner = make_clf(C=C, seed=seed)
    clf_inner.fit(X_tr, y_tr)

    val_proba = clf_inner.predict_proba(X_val)[:, 1]
    thr, val_f1 = best_threshold_from_pr(y_val, val_proba)
    
    clf_full = make_clf(C=C, seed=seed)
    clf_full.fit(X, y)
    return clf_full, thr, val_f1


results = {}
skills = get_unique_skills(dir_, files)

for skill in skills:
    start_states, end_states, all_skill_states, negative_end_skill, \
        negative_end_all, all_other_states = get_start_end_states(dir_, skill, features_dirname='pca_features_512')
    positive_states = end_states
    negative_states = negative_end_all


    X = np.vstack([positive_states, negative_states])
    y = np.hstack([
        np.ones(len(positive_states), dtype=int),
        np.zeros(len(negative_states), dtype=int)
    ])
    X, y = shuffle(X, y, random_state=SEED)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=SEED, stratify=y
    )

    print(f"Skill: {skill}")
    print("train balance:", np.bincount(y_train))
    print("test  balance:",  np.bincount(y_test))

    clf, thr, val_f1 = fit_with_threshold(X_train, y_train, C=1.0, seed=SEED)

    # Inspect probabilities and counts at some thresholds
    proba_test = clf.predict_proba(X_test)[:, 1]
    print("min/max prob:", float(proba_test.min()), float(proba_test.max()))
    for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
        preds_t = (proba_test >= t).astype(int)
        print(t, int(preds_t.sum()))
    print("Chosen threshold (from PR/F1 on val):", thr, " (val F1=", f"{val_f1:.4f}", ")")

    # Evaluate at chosen threshold
    y_pred = (proba_test >= thr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    results[skill] = {
        "threshold": thr,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "val_f1": float(val_f1)
    }

    # Persist model + metadata
    model_path = os.path.join(models_dir, f"{skill}_clf.joblib")
    meta_path = os.path.join(models_dir, f"{skill}_meta.json")
    try:
        dump(clf, model_path)
        with open(meta_path, 'w') as f:
            json.dump({
                'skill': skill,
                'threshold': thr,
                'val_f1': val_f1,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'n_train_pos': int((y_train == 1).sum()),
                'n_train_neg': int((y_train == 0).sum()),
                'n_test_pos': int((y_test == 1).sum()),
                'n_test_neg': int((y_test == 0).sum()),
                'seed': SEED
            }, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save model/metadata for skill {skill}: {e}")

    # print(f"Results: {results[skill]}")
    # print(classification_report(y_test, y_pred, zero_division=0))
    # print("=" * 50 + "\n")

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
        "threshold": res["threshold"],
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
macro_precision = float(np.mean([r["precision"] for r in rows])) if rows else float("nan")
macro_recall    = float(np.mean([r["recall"]    for r in rows])) if rows else float("nan")
macro_f1        = float(np.mean([r["f1"]        for r in rows])) if rows else float("nan")
macro_accuracy  = float(np.mean([r["accuracy"]  for r in rows])) if rows else float("nan")

# Pretty print
print("\n" + "="*80)
print("PER-SKILL METRICS (sorted by F1 desc)")
print("="*80)


df = pd.DataFrame(rows)
df = df.sort_values("f1", ascending=False)
disp_cols = ["skill", "pos_support", "neg_support", "threshold",
                "precision", "recall", "f1", "accuracy", "tp", "fp", "fn"]
# Round numeric columns for readability
for c in ["threshold", "precision", "recall", "f1", "accuracy"]:
    df[c] = df[c].astype(float).round(3)
print(df[disp_cols].to_string(index=False))

# Save per-skill metrics table & overall summary
metrics_csv = os.path.join(models_dir, 'per_skill_metrics.csv')
metrics_json = os.path.join(models_dir, 'summary_metrics.json')
try:
    df.to_csv(metrics_csv, index=False)
    with open(metrics_json, 'w') as f:
        json.dump({
            'overall': {
                'support': overall_support,
                'tp': tot_tp, 'fp': tot_fp, 'fn': tot_fn, 'tn': tot_tn,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'accuracy': overall_accuracy
            },
            'macro': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1,
                'accuracy': macro_accuracy
            }
        }, f, indent=2)
except Exception as e:
    print(f"[WARN] Failed to save aggregate metrics: {e}")

print("\n" + "="*80)
print("OVERALL (MICRO) METRICS — pooled over all skills")
print("="*80)
print(f"Support (all skills): {overall_support}")
print(f"TP={tot_tp}  FP={tot_fp}  FN={tot_fn}  TN={tot_tn}")
print(f"Precision: {overall_precision:.3f}  Recall: {overall_recall:.3f}  F1: {overall_f1:.3f}  Accuracy: {overall_accuracy:.3f}")

print("\n" + "="*80)
print("MACRO AVERAGES — mean of per-skill metrics")
print("="*80)
print(f"Precision: {macro_precision:.3f}  Recall: {macro_recall:.3f}  F1: {macro_f1:.3f}  Accuracy: {macro_accuracy:.3f}")
print("="*80 + "\n")